#!/usr/bin/env Rscript

# Visualise a raw FastAAI matrix as a clustered heatmap with dendrograms.
# Usage:
#   Rscript scripts/visualise_aai_matrix.R [FastAAI_matrix.txt]
#
# FastAAI matrix semantics:
#   0.0  = no shared SCPs
#   15.0 = reported as <30% AAI
#   95.0 = reported as >90% AAI

die <- function(message) {
  # Stop execution with a concise error message.
  stop(message, call. = FALSE)
}

parse_args <- function() {
  # Return the input matrix path from the command line.
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) > 1) {
    die("Usage: Rscript scripts/visualise_aai_matrix.R [FastAAI_matrix.txt]")
  }
  if (length(args) == 0) {
    return("FastAAI_matrix.txt")
  }
  args[[1]]
}

read_matrix_file <- function(matrix_path) {
  # Read and validate a raw FastAAI matrix file.
  if (!file.exists(matrix_path)) {
    die(sprintf("FastAAI matrix file not found: %s", matrix_path))
  }

  lines <- readLines(matrix_path, warn = FALSE, encoding = "UTF-8")
  lines <- lines[nzchar(lines)]
  if (length(lines) == 0) {
    die("FastAAI matrix is empty.")
  }

  fields <- strsplit(lines, "\t", fixed = TRUE)
  header <- fields[[1]]
  if (length(header) < 2) {
    die("FastAAI matrix header must contain 'query_genome' plus at least one genome name.")
  }
  if (header[[1]] != "query_genome") {
    die(sprintf(
      "FastAAI matrix first header cell must be 'query_genome', got: '%s'",
      header[[1]]
    ))
  }

  genome_names <- header[-1]
  if (length(genome_names) < 2) {
    die("FastAAI matrix must contain at least two genomes.")
  }
  if (anyDuplicated(genome_names) > 0) {
    duplicates <- unique(genome_names[duplicated(genome_names)])
    die(sprintf(
      "FastAAI matrix header contains duplicate genome names: %s",
      paste(duplicates, collapse = ", ")
    ))
  }

  matrix_rows <- fields[-1]
  if (length(matrix_rows) != length(genome_names)) {
    die(sprintf(
      "FastAAI matrix must be square: expected %d data rows, found %d.",
      length(genome_names),
      length(matrix_rows)
    ))
  }

  matrix_values <- matrix(
    NA_real_,
    nrow = length(genome_names),
    ncol = length(genome_names),
    dimnames = list(genome_names, genome_names)
  )

  expected_width <- length(genome_names) + 1
  for (row_index in seq_along(matrix_rows)) {
    row_fields <- matrix_rows[[row_index]]
    if (length(row_fields) != expected_width) {
      die(sprintf(
        "FastAAI matrix row %d has %d columns, expected %d.",
        row_index,
        length(row_fields),
        expected_width
      ))
    }

    row_name <- row_fields[[1]]
    expected_name <- genome_names[[row_index]]
    if (row_name != expected_name) {
      die(sprintf(
        "FastAAI matrix row %d name '%s' does not match header name '%s' at the same position.",
        row_index,
        row_name,
        expected_name
      ))
    }

    raw_values <- row_fields[-1]
    numeric_values <- suppressWarnings(as.numeric(raw_values))
    if (any(is.na(numeric_values)) || any(!is.finite(numeric_values))) {
      bad_index <- which(is.na(numeric_values) | !is.finite(numeric_values))[1]
      die(sprintf(
        "Non-numeric FastAAI value at row '%s', column '%s': '%s'",
        row_name,
        genome_names[[bad_index]],
        raw_values[[bad_index]]
      ))
    }
    if (any(numeric_values < 0) || any(numeric_values > 95)) {
      bad_index <- which(numeric_values < 0 | numeric_values > 95)[1]
      die(sprintf(
        "FastAAI value out of the supported raw range [0,95] at row '%s', column '%s': %s",
        row_name,
        genome_names[[bad_index]],
        raw_values[[bad_index]]
      ))
    }

    matrix_values[row_index, ] <- numeric_values
  }

  tolerance <- 1e-8
  symmetric <- abs(matrix_values - t(matrix_values)) <= tolerance
  if (!all(symmetric)) {
    mismatch <- which(!symmetric, arr.ind = TRUE)[1, ]
    die(sprintf(
      paste(
        "FastAAI matrix is not symmetric:",
        "%s vs %s (%s != %s)"
      ),
      rownames(matrix_values)[[mismatch[[1]]]],
      colnames(matrix_values)[[mismatch[[2]]]],
      format(matrix_values[mismatch[[1]], mismatch[[2]]], trim = TRUE),
      format(matrix_values[mismatch[[2]], mismatch[[1]]], trim = TRUE)
    ))
  }

  matrix_values
}

build_distance_matrix <- function(matrix_values) {
  # Convert the observed FastAAI similarity scale into a complete-linkage distance.
  similarity_ceiling <- max(matrix_values)
  distance_values <- similarity_ceiling - matrix_values
  distance_values[distance_values < 0 & distance_values > -1e-8] <- 0
  if (any(distance_values < 0)) {
    die("Derived negative distances from the FastAAI matrix.")
  }
  diag(distance_values) <- 0
  as.dist(distance_values)
}

derive_label_stride <- function(genome_count) {
  # Choose a deterministic thinning interval for axis labels.
  if (genome_count <= 60) {
    return(1)
  }
  if (genome_count <= 150) {
    return(5)
  }
  if (genome_count <= 400) {
    return(10)
  }
  25
}

build_axis_labels <- function(genome_names) {
  # Keep a sparse, predictable subset of labels while always preserving the ends.
  genome_count <- length(genome_names)
  label_stride <- derive_label_stride(genome_count)
  keep_indices <- seq.int(1, genome_count, by = label_stride)
  keep_indices <- sort(unique(c(keep_indices, genome_count)))
  axis_labels <- rep("", genome_count)
  axis_labels[keep_indices] <- genome_names[keep_indices]
  list(
    labels = axis_labels,
    keep_indices = keep_indices,
    stride = label_stride
  )
}

derive_label_cex <- function(genome_count) {
  # Scale axis labels to keep larger matrices readable.
  if (genome_count <= 60) {
    return(0.7)
  }
  if (genome_count <= 150) {
    return(0.5)
  }
  if (genome_count <= 400) {
    return(0.4)
  }
  0.3
}

derive_device_size <- function(genome_count) {
  # Choose a device size that grows with matrix size but stays bounded.
  size_inches <- 10 + genome_count * 0.03
  max(10, min(40, size_inches))
}

draw_heatmap <- function(matrix_values, matrix_label) {
  # Draw a clustered heatmap with matched dendrograms and a FastAAI legend.
  genome_count <- nrow(matrix_values)
  distance_matrix <- build_distance_matrix(matrix_values)
  clustering <- hclust(distance_matrix, method = "complete")
  ordered_indices <- clustering$order
  ordered_matrix <- matrix_values[ordered_indices, ordered_indices, drop = FALSE]
  dendrogram <- as.dendrogram(clustering)
  label_cex <- derive_label_cex(genome_count)
  axis_label_info <- build_axis_labels(colnames(ordered_matrix))
  draw_grid <- genome_count <= 75
  matrix_margin <- if (axis_label_info$stride == 1) 4.5 else 3.5

  palette_values <- grDevices::colorRampPalette(
    c("#f7fbff", "#dbe9f6", "#9ecae1", "#4292c6", "#2171b5", "#084594")
  )(256)
  legend_breaks <- c(0, 15, 30, 60, 90, 95)

  graphics::layout(
    matrix(c(0, 1, 0, 2, 3, 4), nrow = 2, byrow = TRUE),
    widths = c(0.9, 12.0, 1.2),
    heights = c(0.9, 12.0)
  )

  graphics::par(mar = c(0, 0, 0.4, 0))
  graphics::plot.new()

  graphics::par(mar = c(0.2, 0.6, 0.8, 0.2))
  graphics::plot(dendrogram, axes = FALSE, xaxs = "i", leaflab = "none")
  graphics::mtext(matrix_label, side = 3, line = 0.1, cex = 0.75)

  graphics::par(mar = c(0.6, 0.2, 0.2, 0.2))
  graphics::plot(dendrogram, horiz = TRUE, axes = FALSE, yaxs = "i", leaflab = "none")

  graphics::par(mar = c(matrix_margin, matrix_margin, 0.4, 0.4))
  image_values <- t(ordered_matrix[nrow(ordered_matrix):1, , drop = FALSE])
  graphics::image(
    x = seq_len(genome_count),
    y = seq_len(genome_count),
    z = image_values,
    col = palette_values,
    zlim = c(0, 95),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "",
    xaxs = "i",
    yaxs = "i",
    useRaster = TRUE
  )
  graphics::axis(
    side = 1,
    at = seq_len(genome_count),
    labels = axis_label_info$labels,
    las = 2,
    cex.axis = label_cex
  )
  graphics::axis(
    side = 2,
    at = seq_len(genome_count),
    labels = rev(axis_label_info$labels),
    las = 2,
    cex.axis = label_cex
  )
  if (draw_grid) {
    graphics::abline(h = seq(0.5, genome_count + 0.5, by = 1), col = "#ffffff66", lwd = 0.35)
    graphics::abline(v = seq(0.5, genome_count + 0.5, by = 1), col = "#ffffff66", lwd = 0.35)
  }
  graphics::box()

  graphics::par(mar = c(1.6, 0.2, 0.2, 1.8))
  graphics::image(
    x = 1,
    y = seq(0, 95, length.out = length(palette_values)),
    z = matrix(seq(0, 95, length.out = length(palette_values)), nrow = 1),
    col = palette_values,
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = ""
  )
  graphics::axis(side = 4, at = legend_breaks, labels = legend_breaks, las = 2, cex.axis = 0.65)
  graphics::mtext("Raw value", side = 4, line = 1.1, cex = 0.6)
  graphics::mtext("0 = no shared SCPs", side = 1, line = 0.8, cex = 0.5)
  graphics::mtext("15 = <30%, 95 = >90%", side = 1, line = 1.7, cex = 0.5)
}

write_outputs <- function(matrix_values, matrix_path) {
  # Render an SVG heatmap beside the input matrix.
  output_dir <- dirname(matrix_path)
  svg_path <- file.path(output_dir, "FastAAI_matrix_heatmap.svg")
  figure_size <- derive_device_size(nrow(matrix_values))
  matrix_label <- basename(matrix_path)

  grDevices::svg(svg_path, width = figure_size, height = figure_size)
  draw_heatmap(matrix_values, matrix_label)
  grDevices::dev.off()

  message(sprintf("Wrote %s", svg_path))
}

main <- function() {
  # Run the FastAAI matrix visualisation workflow.
  matrix_path <- parse_args()
  matrix_values <- read_matrix_file(matrix_path)
  write_outputs(matrix_values, matrix_path)
  0
}

quit(save = "no", status = main())
