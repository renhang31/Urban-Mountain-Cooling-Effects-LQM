# Load necessary packages
if (!require("terra")) install.packages("terra")
library(terra)
library(readr)
library(dplyr)

# --------------------------
# Path Configuration (EASY TO MODIFY!)
# --------------------------
base_dir <- "C:/Users/admin/Desktop/Cooling_Effects_Large_Urban_Mountains"
input_dir <- file.path(base_dir, "2023") # change your years 
output_dir <- file.path(base_dir, "2023") # change your years 
shapefile_path <- file.path(base_dir, "scope/lqs.shp")
csv_input_path <- file.path(output_dir, "resampledclip_data.csv")
csv_output_path <- file.path(output_dir, "resampledclip_data_sorted_standardized.csv")

# Create output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# --------------------------
# Step 1: Batch Crop Rasters to Vector Extent
# --------------------------
process_rasters_crop <- function(file_path) {
  r <- rast(file_path)
  
  # Reproject if coordinate systems don't match
  if (!identical(crs(r), crs(shape))) {
    message("Reprojecting for cropping: ", basename(file_path))
    r <- project(r, crs(shape))
  }
  
  # Crop raster to shapefile extent
  message("Cropping: ", basename(file_path))
  r <- crop(r, shape)
  
  return(r)
}

# Get all raster files and the shapefile
tif_files <- list.files(input_dir, pattern = "\\.tif$", full.names = TRUE)
shape <- vect(shapefile_path)

# Execute cropping operation
message("Starting cropping process...")
cropped_rasters <- lapply(tif_files, process_rasters_crop)

# --------------------------
# Step 2: Determine Template Raster (Lowest Resolution)
# --------------------------
# Calculate resolutions and find the largest cell size
res_list <- lapply(cropped_rasters, function(r) res(r))
max_area_index <- which.max(sapply(res_list, function(r) prod(r[1], r[2])))
template <- cropped_rasters[[max_area_index]]

message("Using template raster with resolution: ", paste(res(template), collapse = "x"), 
        " (file: ", names(template), ")")

# --------------------------
# Step 3: Batch Resample to Template Resolution
# --------------------------
process_rasters_resample <- function(r) {
  # Reproject if coordinate systems don't match template
  if (!identical(crs(r), crs(template))) {
    message("Reprojecting: ", names(r))
    r <- project(r, crs(template))
  }
  
  # Resample only if resolutions differ
  if (!all(res(r) == res(template))) {
    message("Resampling: ", names(r))
    r <- resample(r, template, method = "bilinear")  # Bilinear for continuous data
  }
  return(r)
}

# Execute resampling
message("Starting resampling process...")
processed_rasters <- lapply(cropped_rasters, process_rasters_resample)
stacked_rasters <- rast(processed_rasters)

# --------------------------
# Step 4: Export to CSV
# --------------------------
# Create data frame with NA removal
df <- as.data.frame(stacked_rasters, na.rm = TRUE)

# Clean column names (remove year suffix)
colnames(df) <- gsub("_2023.*", "", names(stacked_rasters))# change your years

# Write CSV file
write.csv(df, file = csv_input_path, row.names = FALSE)
message("CSV file created: ", csv_input_path)
# --------------------------
# Step 5: Data Standardization and Reordering
# --------------------------
message("Starting data standardization and reordering...")

# Read the raw combined data
df_raw <- read_csv(csv_input_path)

# Dynamically create column order: all columns except 'MCI' first, then 'MCI' at the end
all_columns <- names(df_raw)
columns_except_mci <- setdiff(all_columns, "MCI")
new_column_order <- c(columns_except_mci, "MCI")

# Process data: reorder and standardize
# Use as.vector() to convert the matrix output of scale() back to a simple vector
df_processed <- df_raw %>%
  select(all_of(new_column_order)) %>%  # Reorder columns with MCI at the end
  mutate(across(-MCI, ~as.vector(scale(.x))))  # Standardize and convert to vector

# Export final standardized data
write_csv(df_processed, csv_output_path)
message("Standardized CSV file created: ", csv_output_path, "\n")

# --------------------------
# Verification
# --------------------------
cat("=== Processing Complete Successfully! ===\n")
cat("Original combined data: ", csv_input_path, "\n")
cat("Final standardized data: ", csv_output_path, "\n")
cat("Variables included (in order): ", paste(names(df_processed), collapse = ", "), "\n")

