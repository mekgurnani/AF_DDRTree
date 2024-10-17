library(dplyr)
library(monocle)
library(reticulate)

bidmc_af_subset <- read.csv("../prev_af_ecg_opbidmc.csv")

filtered_bidmc_af_subset <- bidmc_af_subset %>%
  group_by(id_x) %>%
  mutate(ecg_dt = as.Date(ecg_date_x)) %>%
  arrange(ecg_date_x) %>%
  slice(1) %>% 
  ungroup() %>%
  select(id_x, filename, matches("^X[0-9]+$"), age_ecg_x,female_x)

filtered_bidmc_af_subset_cc <- filtered_bidmc_af_subset[complete.cases(filtered_bidmc_af_subset),]

data_45k = filtered_bidmc_af_subset_cc %>% select(
  matches("^X[0-9]+$"), age_ecg_x,
  female_x)

labels_45k = filtered_bidmc_af_subset_cc %>% select(-c(
  matches("^X[0-9]+$"), age_ecg_x,
  female_x))

data_45k_norm = data.frame(scale(data_45k))

expression_mat_df <- data.frame(t(data_45k_norm))
expression_mat <- data.matrix(expression_mat_df)

obs_metadata <- cbind(labels_45k)
rownames(obs_metadata) <- colnames(expression_mat_df)

obs_metadata_df <- data.frame(obs_metadata)

feature_metadata <- data.frame(feature_names = colnames(data_45k), row.names = colnames(data_45k))

cds <- monocle::newCellDataSet(expression_mat,
                               phenoData = AnnotatedDataFrame(obs_metadata_df),
                               featureData = AnnotatedDataFrame(feature_metadata),
                               expressionFamily = uninormal())

# Start timing
start_time <- Sys.time()

cds <- reduceDimension(cds,
                       reduction_method = "DDRTree", norm_method = "none",
                       pseudo_expr = 0, scaling = FALSE, verbose = TRUE,
                       relative_expr = FALSE, ncenter = 2000, maxIter = 100,
                       tol = 1e-6
)

# End timing

cds <- orderCells(cds)

end_time <- Sys.time()

# Calculate the elapsed time
elapsed_time <- end_time - start_time

# Print the elapsed time
print(elapsed_time)

plot <- plot_cell_trajectory(cds, color_by = "State")
ggsave("../prevafecg_op_branch_assignments.png", plot = plot, width = 8, height = 6, dpi = 300)
saveRDS(cds, "../prevafecg_op_cds_ddrtree_obj.rds")

# Once the tree is run and analysed as in DDRTree_outputs, 
# pseudotime was re-ordered to begin from the core tree branch by loading the CDS object 
# and only running the following code to save the new tree object
cds <- readRDS("../prevafecg_op_cds_ddrtree_obj.rds")
cds <- orderCells(cds, root_state = 19)
saveRDS(cds, "../broadQRS_ddrtree_cds_reordered19.rds")


