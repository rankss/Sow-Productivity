library(corrplot)

# Define the dataset name (should match the one used in the Python script)
dataset_name <- "cdpq"

# Construct the file path to the processed CSV
file_path <- paste0("processed_data/", dataset_name, "_processed_dataset.csv")

# Read the CSV file into a data frame
df <- read.csv(file_path)

# Subset the columns from the 3rd to the second-to-last, similar to Python's [2:-2]
# In R, we specify the range of columns to select.
num_cols <- ncol(df)
X <- df[, 3:(num_cols - 1)]

# Print the first few rows to verify it loaded correctly

print(head(X))

# --- Create and save a correlation plot ---

# Calculate the correlation matrix for the numeric feature set
correlation_matrix <- cor(X)

# Define the output path for the plot, relative to the script's location
figure_directory <- paste0("figures/", dataset_name, "/")
# Ensure the directory exists
dir.create(figure_directory, showWarnings = FALSE, recursive = TRUE)
plot_path <- paste0(figure_directory, "r_correlation_plot.png")

# Open a PNG device to save the plot with high resolution
png(filename = plot_path, width = 13, height = 13, units = "in", res = 300)

# Create a mixed correlation plot
corrplot.mixed(correlation_matrix,
               lower = "number", # Show correlation coefficients in the lower triangle
               upper = "circle", # Show circles in the upper triangle
               tl.col = "black", # Text label color
               tl.srt = 45)      # Rotate text labels for better readability

# Close the device, which saves the file
dev.off()

print(paste("Correlation plot saved to:", plot_path))
