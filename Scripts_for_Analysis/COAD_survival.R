library(readr)
library(dplyr)
library(tidyverse)
library(stringr)
library(survival)
library(survminer)
library(ggplot2)
library(tidyr)

library(GGally)
library(RColorBrewer)


## Read and preprocess COAD dataset --------------------------------------

df <- read.csv("COAD_Distance_Clinical_Merged.csv")

df <- df %>%
  separate(
    col = Sample_ID,
    into = c("Project", "TSS", "Participant", "SampleType_Combined", "Portion", "SlideID"),
    sep = "-",
    remove = FALSE,
    fill = "right"
  ) %>%
  mutate(
    patient = paste("TCGA", TSS, Participant, sep = "-"),
    Stage_Group = case_when(
      ajcc_pathologic_tumor_stage %in% 
        c("Stage I", "Stage IA", "Stage IB") ~ "Stage_I",
      ajcc_pathologic_tumor_stage %in% c("Stage II", "Stage IIA", "Stage IIB") ~ "Stage_II",
      ajcc_pathologic_tumor_stage %in% c("Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC") ~ "Stage_III",
      ajcc_pathologic_tumor_stage %in% c("Stage IV") ~ "Stage_IV",
      TRUE ~ NA_character_
    ),
    Stage_Group = factor(Stage_Group,
                         levels = c("Stage_I","Stage_II","Stage_III","Stage_IV"))
  ) %>%
  filter(!is.na(Stage_Group)) %>%
  select(-ajcc_pathologic_tumor_stage) %>%
  rename(Age = age_at_initial_pathologic_diagnosis)

subjects <- unique(df$Participant)

# Variable Setup (Numeric & Cont.) ----------------------------------------

# Define Continuous Variables
cont_vars <- names(df)[sapply(df, is.numeric)]
cont_vars <- setdiff(cont_vars, c("PFI.time", "OS.time", "PFI", "OS"))

length(cont_vars)
cont_vars

df_long_cont <- df %>%
  select(all_of(cont_vars)) %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")


# Distance & Proportion Patterns
distance_vars <- grep(".*_to_.*_(mean|std|median|iqr)$", names(df), value = TRUE)
dist_median <- grep(".*_to_.*_median$", names(df), value = TRUE)
dist_iqr <- grep(".*_to_.*_iqr$", names(df), value = TRUE)
dist_mean <- grep(".*_to_.*_mean$", names(df), value = TRUE)
dist_std  <- grep(".*_to_.*_std$",  names(df), value = TRUE)

prop_vars <- grep("_overall_proportion$", colnames(df), value = TRUE)
rowSums(df[,  prop_vars])

demo_vars <- c("Age", "gender")
strata_vars <- c("Stage_Group")



# Visualizations ----------------------------------------------------------

#### Proportions: KM tertiles + histograms  ####

# Histogram with cutpoints
cutpoints <- map_dfr(prop_vars, function(v) {
  x <- df[[v]]
  qs <- quantile(x, probs = c(1/3, 2/3), na.rm = TRUE)
  tibble(
    variable = v, cut1 = qs[1], cut2 = qs[2],
    n_low = sum(x < qs[1], na.rm = TRUE),
    n_median = sum(x >= qs[1] & x < qs[2], na.rm = TRUE),
    n_high = sum(x >= qs[2], na.rm = TRUE)
  )
})

cutpoints

p_prop_hist <- ggplot(
  df_long_cont %>% filter(variable %in% prop_vars)  , 
  aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_bw(base_size = 11) +
  geom_vline(aes(xintercept = cut1), data = cutpoints, linetype = "dashed", color = "blue") +
  geom_vline(aes(xintercept = cut2), data = cutpoints, linetype = "dashed", color = "blue") +
  labs(
    title = paste("Histograms: Overall Proportion"),
    x = "Value",
    y = "Count"
  )

ggsave("COAD_histograms_proportions.jpg", height = 5, width = 8)


# Create tertile groups for proportions
df_tert <- df
for (v in prop_vars) {
  qs <- quantile(df_tert[[v]], probs = c(1/3, 2/3), na.rm = TRUE)
  
  new_name <- paste0(v, "_tert")  # e.g. Cancer_overall_proportion_tert
  
  df_tert[[new_name]] <- cut(
    df_tert[[v]],
    breaks = c(-Inf, qs[1], qs[2], Inf),
    labels = c("Low", "Medium", "High"),
    include.lowest = TRUE
  )
}

# KM PFI by tertile groups for proportions
prop_tert_vars <- paste0(prop_vars, "_tert")
p_prop_list <- list()

for (var in prop_tert_vars) {
  f <- paste0("Surv(PFI.time, PFI) ~ ", var)
  fit <- survfit(as.formula(f), data = df_tert)
  base_name <- gsub("_overall_proportion_tert$", "", var)
  
  p1 <- ggsurvplot(
    fit,
    data = df_tert,
    pval = TRUE,
    palette = c("green", "blue", "orange", "red"),
    risk.table = FALSE,
    legend.title = paste0(base_name),
    legend.labs = levels(df_tert[[var]]),
    xlab = "Time (days)",
    ylab = "Progression-Free Survival Probability",
    ggtheme = theme_bw(base_size = 18)
  )
  p_prop_list[[var]] <- p1
}

combined_prop <- ggarrange(
  plotlist = lapply(p_prop_list[prop_tert_vars], 
                    function(x) x$plot),
  nrow = 2,
  ncol = ceiling(length(prop_tert_vars) / 2)
) +
  plot_annotation(
    title = "KM: PFI by Overall Proportion Groups",
    theme = theme(plot.title = element_text(hjust = 0.5, size = 22, face = "bold") )
  )

ggsave("COAD_KM_PFI_by_Overall_Proportion_Groups.png", combined_prop, width = 16, height = 12, dpi = 300)



#### Distance: KM binary cut at median + histograms ####

# Histogram with binary cutpoints at median
dist_median_cuts <- map_dfr(dist_median, function(v) {
  x <- df[[v]]
  m <- median(x, na.rm = TRUE)
  
  tibble(
    variable = v,
    cutpoint_median = m,
    n_low = sum(x <= m, na.rm = TRUE),
    n_high = sum(x >  m, na.rm = TRUE)
  )
})

dist_median_cuts

p_dist_hist <- ggplot(
  df_long_cont %>% filter(variable %in% dist_median), 
       aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_bw(base_size = 11) +
  geom_vline(aes(xintercept = cutpoint_median), data = dist_median_cuts, linetype = "dashed", color = "blue") +
  labs(
    title = paste("Histograms: Median Distance"),
    x = "Value",
    y = "Count"
  )

ggsave("COAD_histograms_median_distance.jpg", width = 10, height = 10 )

# Create binary groups for median distance
df <- df %>%
  mutate(
    across(
      all_of(dist_median),
      ~ cut(.x, breaks = c(-Inf, median(.x, na.rm = TRUE), Inf),
        labels = c("Low", "High"), include.lowest = TRUE),
      .names = "{.col}_bin"
    ))

# KM by median distance (Low vs High) 
distance_bin_vars <- paste0(dist_median, "_bin")

p_dist_median_list <- list()
for (var in distance_bin_vars) {
  f <- paste0("Surv(PFI.time, PFI) ~ ", var)
  fit <- survfit(as.formula(f), data = df)
  base_name <- gsub("_(mean|median|overall_proportion)_bin$", "", var)
  
  p1 <- ggsurvplot(
    fit, data = df, pval = TRUE,
    palette = c("green", "blue", "orange", "red"),
    risk.table = FALSE, legend.title = base_name,
    legend.labs = levels(df[[var]]),
    xlab = "Time (days)", ylab = "Progression-Free Survival Probability",
    ggtheme = theme_bw(base_size = 18)
  )
  p_dist_median_list[[var]] <- p1
}

combined_median <- ggarrange(
  plotlist = lapply(p_dist_median_list, function(x) x$plot),
  ncol = 3,
  nrow = ceiling(length(distance_bin_vars) / 3)
) +
  plot_annotation(
    title = "KM: PFI by Median Distance Groups",
    theme = theme(plot.title = element_text(hjust = 0.5, size = 30, face = "bold"))
  )

ggsave("COAD_KM_PFI_by_Distance_All_Median_Groups.png", combined_median, width = 16, height =24, dpi = 300)

#### Tumor → APC distance: KM PFI stratified by Stage ####

# Specify the Tumor → APC median distance binary variable

df <- df %>%
  mutate(Stage =  case_when(
    Stage_Group %in% 
      c("Stage_II", "Stage_III") ~ "Stage II or III",
    Stage_Group == "Stage_I" ~ "Stage I",
    Stage_Group == "Stage_IV" ~ "Stage IV",
    TRUE ~ NA_character_
  ))

# Fit KM with Tumor→APC (Low/High) and Stage_Group
fit_pfi_tumor_apc_stage <- survfit(
  Surv(PFI.time, PFI) ~ Cancer_to_pan.APC_median_bin + Stage,
  data = df
)

# Plot: one panel per stage, lines = Low vs High Tumor→APC distance
p_pfi_tumor_apc_stage <- ggsurvplot_facet(
  fit_pfi_tumor_apc_stage,
  data = df,
  facet.by = "Stage",
  legend.title = "Cancer_to_pan.APC Median Distance",
  legend.labs  = levels(df[[tumor_apc_bin]]),
  pval = TRUE,
  risk.table = FALSE,
  xlab = "Time (days)",
  ylab = "Progression-Free Survival Probability",
  ggtheme = theme_bw(base_size = 16),
  palette = c("blue", "orange")
)

ggsave(
  "COAD_KM_PFI_Tumor_to_APC_by_Stage.png",
  p_pfi_tumor_apc_stage$plot,
  width = 10, height = 8, dpi = 300
)



#### Tumor → APC distance: KM PFI, top 1/3 vs bottom 1/3, stratified by Stage ####

tumor_apc_var <- "Cancer_to_pan.APC_median" 
tumor_apc_qs <- quantile(df[[tumor_apc_var]], probs = c(1/3, 2/3), na.rm = TRUE)

df_tumor_apc_tert <- df %>%
  mutate(
    Cancer_to_pan.APC_median_tert = cut(
      .data[[tumor_apc_var]],
      breaks = c(-Inf, tumor_apc_qs[1], tumor_apc_qs[2], Inf),
      labels = c("Bottom 1/3", "Middle 1/3", "Top 1/3"),
      include.lowest = TRUE
    )
  ) %>%
  # Keep only bottom vs top third
  filter(Cancer_to_pan.APC_median_tert != "Middle 1/3") %>%
  droplevels()

table(df_tumor_apc_tert$Cancer_to_pan.APC_median_tert, df_tumor_apc_tert$Stage_Group)

# KM fit: Tumor→APC tertiles (Bottom vs Top) + Stage_Group
fit_pfi_tumor_apc_tert_stage <- survfit(
  Surv(PFI.time, PFI) ~ Cancer_to_pan.APC_median_tert + Stage,
  data = df_tumor_apc_tert
)

# Plot: one facet per stage, lines = Bottom 1/3 vs Top 1/3
p_pfi_tumor_apc_tert_stage <- ggsurvplot_facet(
  fit_pfi_tumor_apc_tert_stage,
  data = df_tumor_apc_tert,
  facet.by    = "Stage",
  legend.title = "Cancer_to_pan.APC Median Distance",
  legend.labs  = levels(df_tumor_apc_tert$Cancer_to_pan.APC_median_tert),
  pval        = TRUE,
  risk.table  = FALSE,
  xlab        = "Time (days)",
  ylab        = "Progression-Free Survival Probability",
  ggtheme     = theme_bw(base_size = 16),
  palette     = c("blue", "orange")  # Bottom 1/3, Top 1/3
)

ggsave(
  "COAD_KM_PFI_Tumor_to_APC_Tertiles_by_Stage.png",
  p_pfi_tumor_apc_tert_stage$plot,
  width = 10, height = 8, dpi = 300
)














#### Age: KM quartiles + histogram ####

ggplot(df_long_cont %>% filter(variable == "Age")  , 
       aes(x = value)) +
  geom_histogram(bins = 40, fill = "grey30", color = "white") +
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  theme_bw(base_size = 11) +
  labs(
    title = paste("Histograms: Age"),
    x = "Value",
    y = "Count"
  )

ggsave("COAD_histograms_age.jpg", width = 3, height = 3)


# Create quartiles for age
age_cuts <- quantile(df$Age, probs = c(0, 0.25, 0.50, 0.75, 1), na.rm = TRUE)
age_cuts

df <- df %>%
  mutate(
    Age_quart = cut(
      Age,
      breaks = age_cuts,
      include.lowest = TRUE,
      labels = c("Q1", "Q2", "Q3", "Q4")
    )
  )

df$Age_quart <- factor(df$Age_quart)

table(df$Age_quart)


# KM by age quartiles
fit_age <- survfit(Surv(PFI.time, PFI) ~ Age_quart, data=df)

png(paste0("COAD_KM_PFI_by_age_quart.png"), width = 500, height = 500)

p1 <- ggsurvplot(
  fit_age,
  data=df,
  pval=TRUE,
  palette=c("green", "blue", "orange", "red"),
  risk.table=FALSE,
  legend.title="Age Quartiles", 
  legend.labs = levels(df[["Age_quart"]]),
  xlab="Time (days)",
  ylab="Progression-Free Survival Probability",
  ggtheme=theme_bw(base_size=18),
  combine = TRUE
)

print(p1)

dev.off()



#### Stage, Gender: KM  ####

# KM by Stage
table(df$Stage_Group)
fit_pfi_stage <- survfit(Surv(PFI.time, PFI) ~ Stage_Group, data=df)

png("COAD_KM_PFI_by_Stage.png", width = 500, height = 500)
ggsurvplot(
  fit_pfi_stage,
  data=df,
  palette=c("green","blue","orange","red"),
  pval=TRUE,
  conf.int=FALSE,
  legend.title="",
  legend.labs = levels(df$Stage_Group),
  risk.table=FALSE,
  xlab="Time (days)",
  ylab="Progression-Free Survival Probability",
  ggtheme=theme_bw(base_size=18),
  combine = TRUE
)
dev.off()


# KM by Gender
table(df$gender)
fit_pfi_gender <- survfit(Surv(PFI.time, PFI) ~ gender, data=df)

png("COAD_KM_PFI_by_gender.png", width = 500, height = 500)
ggsurvplot(
  fit_pfi_gender,
  data=df,
  palette=c("green","blue","orange","red"),
  pval=TRUE,
  conf.int=FALSE,
  legend.title="",
  legend.labs = levels(df$race),
  risk.table=FALSE,
  xlab="Time (days)",
  ylab="Progression-Free Survival Probability",
  ggtheme=theme_bw(base_size=18),
  combine = TRUE
)
dev.off()



# Correlations ------------------------------------------------------------

#### Overall correlation heatmap ####

# Correlation matrix over distance + proportions
all_vars <- c(distance_vars, prop_vars)
cor_all <- cor(df[, all_vars],
               use = "pairwise.complete.obs",
               method = "pearson")

# Create heatmap
distance_stem <- sub("_(mean|median|std|iqr)$", "", distance_vars)
cell_from <- sub("_to_.*", "", distance_stem)
cell_to <- sub(".*_to_", "", distance_stem)

pair_id <- ifelse(cell_from < cell_to,
                  paste(cell_from, cell_to, sep = "_"),
                  paste(cell_to, cell_from, sep = "_"))
prop_pair_id <- rep("Overall_Proportion", length(prop_vars))

# Combined in the same order as cor_all rows/cols
pair_vec <- c(pair_id, prop_pair_id)
pair_fac <- factor(pair_vec, levels = unique(pair_vec))
pair_levels <- levels(pair_fac)
n_pairs <- length(pair_levels)
pair_colors <- structure(brewer.pal(min(n_pairs, 8), "Set2"), names = pair_levels)

top_ann <- HeatmapAnnotation(
  Pair = pair_fac,
  col  = list(Pair = pair_colors),
  
  show_annotation_name = FALSE,
  annotation_name_gp   = gpar(fontsize = 0),
  
  show_legend = TRUE,
  annotation_legend_param = list(
    title_gp  = gpar(fontsize = 8, fontface = "bold"),
    labels_gp = gpar(fontsize = 6)
  )
)

right_ann <- rowAnnotation(
  Pair = pair_fac,
  col = list(Pair = pair_colors),
  show_legend = FALSE,
  show_annotation_name = FALSE
)

ht <- Heatmap(
  cor_all,
  name = "corr",
  col = colorRamp2(c(-1, 0, 1), c("blue", "white", "red")),
  na_col = "white",
  cell_fun = NULL,
  
  # Block structure
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  row_split = pair_fac,
  column_split = pair_fac,
  
  heatmap_width = unit(18, "cm"),
  heatmap_height = unit(18, "cm"),
  row_title  = NULL,
  column_title = NULL,
  row_gap = unit(1, "mm"),
  column_gap = unit(1, "mm"),
  rect_gp = gpar(col = NA),
  border = FALSE,
  show_row_dend = FALSE,
  show_column_dend = FALSE,
  
  show_row_names = TRUE,
  show_column_names = TRUE,
  row_names_side = "left",
  column_names_rot = 90,
  row_names_gp = gpar(fontsize = 6),
  column_names_gp = gpar(fontsize = 6),
  
  heatmap_legend_param = list(
    title = "corr",
    at = c(-1, -0.5, 0, 0.5, 1),
    color_bar = "continuous"
  ),
  
  top_annotation = top_ann,
  right_annotation = right_ann
)

pdf("COAD_corr_all_heatmap.pdf", width = 14, height = 10)
draw(
  ht,
  padding = unit(c(5, 5, 5, 5), "mm"),
  column_title = "Correlation: Distances + Proportions (Undirected Pair Blocks)",
  column_title_gp = gpar(fontsize = 12, fontface = "bold")
)
dev.off()


# GGpairs
outdir <- "COAD_Directional_GGpairs_Facets"
dir.create(outdir, showWarnings = FALSE)

dir_stem <- sub("_(mean|median|std|iqr)$", "", distance_vars)
dir_groups <- split(distance_vars, dir_stem)

for (pair in names(dir_groups)) {
  
  vars <- dir_groups[[pair]]
  
  df_sub <- df[, vars, drop = FALSE]
  df_sub <- df_sub[, sapply(df_sub, is.numeric), drop = FALSE]
  
  if (ncol(df_sub) < 2) next
  
  colnames(df_sub) <- sub(".*_(mean|median|std|iqr)$", "\\1", vars)
  
  p <- ggpairs(
    df_sub,
    title = paste("Directional Pair:", pair),
    progress = FALSE
  )
  
  ggsave(file.path(outdir, paste0(pair, "_ggpairs.png")),
         p, width = 5, height = 5, dpi = 180)
}


# Proportions ILR ------------------------------------------------------


table(df$Cancer_overall_proportion_quart)

library(compositions)
# 5-part composition matrix
prop_mat <- df[, prop_vars]

# Convert to acomp
X <- acomp(prop_mat)

# SBP matrix: rows = parts, cols = balances (ILR coordinates)
## Rows = parts (in same order as prop_vars)
## Cols = ILR balances:
##   ILR1: Tumor (Cancer + Stromal)   vs Non-tumor (APC + T + Normal)
##   ILR2: Cancer                     vs Stromal
##   ILR3: Immune (APC + T cells)     vs Normal epithelium
##   ILR4: T cells                    
sbp <- matrix(
  c(
    # ILR1  ILR2  ILR3  ILR4
    1,     1,    0,    0,   # Cancer
    1,    -1,    0,    0,   # Stromal
    -1,     0,    1,   -1,   # APC
    -1,     0,    1,    1,   # T cells
    -1,     0,   -1,    0    # Normal
  ),
  nrow = length(prop_vars),
  byrow = TRUE
)

rownames(sbp) <- prop_vars
colnames(sbp) <- c(
  "Tumor_vs_NonTumor",
  "Cancer_vs_Stromal",
  "Immune_vs_Normal",
  "Tcells_vs_APC"
)

# Compute ILR with this custom basis
ilr_mat <- ilr(X, V = sbp)

ilr_df <- as.data.frame(ilr_mat)
ilr_vars <- c("ILR_Tumor_vs_NonTumor",
              "ILR_Cancer_vs_Stromal",
              "ILR_Immune_vs_Normal",
              "ILR_Tcells_vs_APC")
names(ilr_df) <- ilr_vars

## Attach back to main df
df <- bind_cols(df, ilr_df)

plot_df <- df %>%
  select(all_of(ilr_vars), Stage_Group)

ggpairs(
  plot_df,
  columns = 1:4,
  aes(colour = Stage_Group, alpha = 0.7),
  upper = list(continuous = "cor"),
  lower = list(continuous = "points")
) +
  theme_bw() +
  theme(legend.position = "bottom")

ggsave("COAD_ILRs_ggpairs.jpg", height = 8, width = 8)

cox_ilr_pfi <- coxph(
  Surv(PFI.time, PFI) ~
    ILR_Tumor_vs_NonTumor +
    ILR_Cancer_vs_Stromal +
    ILR_Immune_vs_Normal +
    ILR_Tcells_vs_APC +
    Age + gender +
    strata(Stage_Group),
  data = df
)


# Marginal Cox Models (PFI) -----------------------------------------------

cox_covars <- c(demo_vars, dist_median, distance_bin_vars, ilr_vars)
margin_cox_vars <- c(strata_vars, cox_covars)

marginal_pfi_results <- list()

for (v in margin_cox_vars) {
  formula <- as.formula(paste0("Surv(PFI.time, PFI) ~ ", v))
  fit <- coxph(formula, data=df)
  
  s <- summary(fit)
  
  marginal_pfi_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}

marginal_pfi_df <- bind_rows(marginal_pfi_results) %>%
  # arrange(pval)  %>%
  # mutate(signif = pval <= 0.05) %>%
  remove_rownames()

print(marginal_pfi_df)

write.csv(marginal_pfi_df, "COAD_marginal_coxph_pfi_results.csv")


# Marginal Cox Models (OS) ------------------------------------------------

marginal_os_results <- list()

for (v in margin_cox_vars) {
  formula <- as.formula(paste0("Surv(OS.time, OS) ~ ", v))
  fit <- coxph(formula, data=df)
  
  s <- summary(fit)
  
  marginal_os_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}

marginal_os_df <- bind_rows(marginal_os_results) %>%
  # arrange(pval) %>%
  # mutate(signif = pval <= 0.05) %>%
  remove_rownames() 

print(marginal_os_df)
write_csv(marginal_os_df, "COAD_marginal_coxph_os_results.csv")


# Adjusted Cox Models (PFI) ---------------------------------------

adjusted_vars <- c(dist_median, distance_bin_vars)

adjusted_pfi_results <- list()

for (v in adjusted_vars) {
  
  fml <- as.formula(
    paste0(
      "Surv(PFI.time, PFI) ~ ",
      v,
      " + Age + gender + Stage_Group"
    )
  )
  
  fit <- coxph(fml, df)
  s <- summary(fit)
  
  adjusted_pfi_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}


adjusted_pfi_df <- bind_rows(adjusted_pfi_results) %>%
  # arrange(pval) %>%
  # remove_rownames() %>%
  mutate(signif = pval <= 0.05)

print(adjusted_pfi_df)

write_csv(adjusted_pfi_df, "COAD_adjusted_marginal_coxph_pfi_results.csv")


adjusted_os_results <- list()

for (v in adjusted_vars) {
  
  fml <- as.formula(
    paste0(
      "Surv(OS.time, OS) ~ ",
      v,
      " + Age + gender + Stage_Group"
    )
  )
  
  fit <- coxph(fml, df)
  s <- summary(fit)
  
  adjusted_os_results[[v]] <- data.frame(
    variable = v,
    variable1 = rownames(s$coefficients),
    HR = s$coefficients[,"exp(coef)"],
    lower95 = s$conf.int[,"lower .95"],
    upper95 = s$conf.int[,"upper .95"],
    pval = s$coefficients[,"Pr(>|z|)"],
    c_index = s$concordance[1]
  )
}


adjusted_os_df <- bind_rows(adjusted_os_results) %>%
  # arrange(pval) %>%
  # remove_rownames() %>%
  mutate(signif = pval <= 0.05)

print(adjusted_os_df)

write_csv(adjusted_os_df, "COAD_adjusted_marginal_coxph_os_results.csv")

