#!/usr/bin/env Rscript

# Visualise a raw FastAAI matrix as paired SVG and PNG heatmaps.
# Usage:
#   Rscript scripts/visualise_aai_matrix.R [--lower-threshold 30] [--upper-threshold 90] [FastAAI_matrix.txt]
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
  # Return the input matrix path and heatmap thresholds from the command line.
  args <- commandArgs(trailingOnly = TRUE)

  parsed <- list(
    matrix_path = "FastAAI_matrix.txt",
    lower_threshold = 30,
    upper_threshold = 90
  )

  index <- 1
  while (index <= length(args)) {
    arg <- args[[index]]
    if (arg == "--lower-threshold") {
      if (index == length(args)) {
        die("Missing value after --lower-threshold")
      }
      parsed$lower_threshold <- parse_threshold_value(args[[index + 1]], "--lower-threshold")
      index <- index + 2
      next
    }
    if (arg == "--upper-threshold") {
      if (index == length(args)) {
        die("Missing value after --upper-threshold")
      }
      parsed$upper_threshold <- parse_threshold_value(args[[index + 1]], "--upper-threshold")
      index <- index + 2
      next
    }
    if (startsWith(arg, "--")) {
      die(sprintf("Unknown option: %s", arg))
    }
    if (parsed$matrix_path != "FastAAI_matrix.txt") {
      die(
        paste(
          "Usage: Rscript scripts/visualise_aai_matrix.R",
          "[--lower-threshold 30] [--upper-threshold 90] [FastAAI_matrix.txt]"
        )
      )
    }
    parsed$matrix_path <- arg
    index <- index + 1
  }

  validate_thresholds(parsed$lower_threshold, parsed$upper_threshold)
  parsed
}

parse_threshold_value <- function(raw_value, option_name) {
  # Parse a user-supplied heatmap threshold as a numeric value.
  numeric_value <- suppressWarnings(as.numeric(raw_value))
  if (length(numeric_value) != 1 || is.na(numeric_value) || !is.finite(numeric_value)) {
    die(sprintf("Unable to parse %s value as a number: %s", option_name, raw_value))
  }
  numeric_value
}

validate_thresholds <- function(lower_threshold, upper_threshold) {
  # Ensure the heatmap scale thresholds are ordered and within the FastAAI range.
  if (lower_threshold < 0 || lower_threshold > 95) {
    die(sprintf("Lower threshold must be within [0,95], got: %s", lower_threshold))
  }
  if (upper_threshold < 0 || upper_threshold > 95) {
    die(sprintf("Upper threshold must be within [0,95], got: %s", upper_threshold))
  }
  if (lower_threshold >= upper_threshold) {
    die(
      sprintf(
        "Lower threshold must be smaller than upper threshold, got: %s >= %s",
        lower_threshold,
        upper_threshold
      )
    )
  }
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

derive_device_size <- function(genome_count, simple = FALSE) {
  # Choose a square SVG size that keeps the matrix visible.
  if (simple) {
    size_inches <- 8 + genome_count * 0.025
    return(max(8, min(32, size_inches)))
  }
  size_inches <- 10 + genome_count * 0.03
  max(10, min(40, size_inches))
}

build_palette <- function(lower_threshold, upper_threshold) {
  # Return the shared FastAAI colour palette and legend ticks.
  midpoint <- round((lower_threshold + upper_threshold) / 2, 2)
  list(
    colours = grDevices::colorRampPalette(
      c("#f7fbff", "#dbe9f6", "#9ecae1", "#4292c6", "#2171b5", "#084594")
    )(256),
    breaks = sort(unique(c(lower_threshold, midpoint, upper_threshold))),
    lower_threshold = lower_threshold,
    upper_threshold = upper_threshold
  )
}

draw_matrix_tiles <- function(
  matrix_values,
  palette_values,
  lower_threshold,
  upper_threshold,
  show_grid = FALSE
) {
  # Draw the matrix as SVG-safe vector tiles.
  genome_count <- nrow(matrix_values)
  image_values <- t(matrix_values[nrow(matrix_values):1, , drop = FALSE])
  graphics::image(
    x = seq_len(genome_count),
    y = seq_len(genome_count),
    z = image_values,
    col = palette_values,
    zlim = c(lower_threshold, upper_threshold),
    xaxt = "n",
    yaxt = "n",
    xlab = "",
    ylab = "",
    xaxs = "i",
    yaxs = "i",
    useRaster = FALSE
  )
  if (show_grid) {
    graphics::abline(h = seq(0.5, genome_count + 0.5, by = 1), col = "#ffffff66", lwd = 0.35)
    graphics::abline(v = seq(0.5, genome_count + 0.5, by = 1), col = "#ffffff66", lwd = 0.35)
  }
  graphics::box()
}

draw_legend <- function(
  palette_values,
  legend_breaks,
  lower_threshold,
  upper_threshold,
  compact = FALSE,
  height_fraction = 0.3,
  top_padding = 0.03
) {
  # Draw a compact boxed legend with an explicit height independent of the matrix panel.
  legend_height <- min(max(height_fraction, 0.1), 0.95)
  legend_top <- min(max(1.0 - top_padding, legend_height), 1.0)
  legend_bottom <- legend_top - legend_height
  legend_edges <- seq(legend_bottom, legend_top, length.out = length(palette_values) + 1)
  legend_left <- 0.48
  legend_right <- 0.72

  graphics::plot.new()
  graphics::plot.window(xlim = c(0, 1), ylim = c(0, 1), xaxs = "i", yaxs = "i")
  graphics::rect(
    xleft = legend_left,
    ybottom = head(legend_edges, -1),
    xright = legend_right,
    ytop = tail(legend_edges, -1),
    col = palette_values,
    border = NA
  )
  graphics::rect(
    xleft = legend_left,
    ybottom = legend_bottom,
    xright = legend_right,
    ytop = legend_top,
    border = "#303030",
    lwd = 0.8
  )
  tick_positions <- legend_bottom + ((legend_breaks - lower_threshold) / (upper_threshold - lower_threshold)) * legend_height
  graphics::axis(side = 2, at = tick_positions, labels = legend_breaks, las = 2, cex.axis = 0.6)
  graphics::mtext("Raw value", side = 2, line = 2.2, cex = 0.55)
  if (compact) {
    graphics::mtext(
      sprintf("<= %.2f uses low colour", lower_threshold),
      side = 1,
      line = 0.5,
      cex = 0.45
    )
    graphics::mtext(
      sprintf(">= %.2f uses high colour", upper_threshold),
      side = 1,
      line = 1.3,
      cex = 0.45
    )
  } else {
    graphics::mtext(
      sprintf("Values <= %.2f use the low colour", lower_threshold),
      side = 1,
      line = 0.6,
      cex = 0.55
    )
    graphics::mtext(
      sprintf("Values >= %.2f use the high colour", upper_threshold),
      side = 1,
      line = 1.6,
      cex = 0.55
    )
  }
}

draw_simple_heatmap <- function(matrix_values, lower_threshold, upper_threshold) {
  # Draw a matrix-only diagnostic heatmap with no labels or dendrograms.
  palette <- build_palette(lower_threshold, upper_threshold)

  graphics::layout(matrix(c(1, 2), nrow = 1), widths = c(1.6, 18))

  graphics::par(mar = c(1.4, 1.8, 0.3, 0.2))
  draw_legend(
    palette$colours,
    palette$breaks,
    palette$lower_threshold,
    palette$upper_threshold,
    compact = TRUE,
    height_fraction = 0.26,
    top_padding = 0.04
  )

  graphics::par(mar = c(0.3, 0.2, 0.3, 0.3))
  draw_matrix_tiles(
    matrix_values,
    palette$colours,
    palette$lower_threshold,
    palette$upper_threshold,
    show_grid = FALSE
  )
}

draw_clustered_heatmap <- function(matrix_values, matrix_label, lower_threshold, upper_threshold) {
  # Draw a clustered heatmap with dendrograms and thinned labels.
  genome_count <- nrow(matrix_values)
  distance_matrix <- build_distance_matrix(matrix_values)
  clustering <- hclust(distance_matrix, method = "complete")
  ordered_indices <- clustering$order
  ordered_matrix <- matrix_values[ordered_indices, ordered_indices, drop = FALSE]
  dendrogram <- as.dendrogram(clustering)
  label_cex <- derive_label_cex(genome_count)
  axis_label_info <- build_axis_labels(colnames(ordered_matrix))
  matrix_margin <- if (axis_label_info$stride == 1) 4.5 else 3.5
  show_grid <- genome_count <= 75
  palette <- build_palette(lower_threshold, upper_threshold)

  graphics::layout(
    matrix(c(0, 1, 0, 4, 2, 3), nrow = 2, byrow = TRUE),
    widths = c(1.3, 0.9, 12.0),
    heights = c(0.9, 12.0)
  )

  graphics::par(mar = c(0, 0, 0.4, 0))
  graphics::plot.new()

  graphics::par(mar = c(0.2, 0.6, 0.8, 0.2))
  graphics::plot(dendrogram, axes = FALSE, xaxs = "i", leaflab = "none")
  graphics::mtext(matrix_label, side = 3, line = 0.1, cex = 0.75)

  graphics::par(mar = c(0.6, 0.2, 0.2, 0.2))
  graphics::plot(dendrogram, horiz = TRUE, axes = FALSE, yaxs = "i", leaflab = "none")

  graphics::par(mar = c(1.4, 1.8, 0.2, 0.1))
  draw_legend(
    palette$colours,
    palette$breaks,
    palette$lower_threshold,
    palette$upper_threshold,
    compact = TRUE,
    height_fraction = 0.28,
    top_padding = 0.04
  )

  graphics::par(mar = c(matrix_margin, matrix_margin, 0.4, 0.3))
  draw_matrix_tiles(
    ordered_matrix,
    palette$colours,
    palette$lower_threshold,
    palette$upper_threshold,
    show_grid = show_grid
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
}

write_outputs <- function(matrix_values, matrix_path, lower_threshold, upper_threshold) {
  # Render clustered and matrix-only SVG and PNG outputs beside the input matrix.
  output_dir <- dirname(matrix_path)
  clustered_svg_path <- file.path(output_dir, "FastAAI_matrix_heatmap.svg")
  simple_svg_path <- file.path(output_dir, "FastAAI_matrix_heatmap_simple.svg")
  clustered_png_path <- file.path(output_dir, "FastAAI_matrix_heatmap.png")
  simple_png_path <- file.path(output_dir, "FastAAI_matrix_heatmap_simple.png")
  matrix_label <- basename(matrix_path)
  clustered_size <- derive_device_size(nrow(matrix_values), simple = FALSE)
  simple_size <- derive_device_size(nrow(matrix_values), simple = TRUE)

  grDevices::svg(
    clustered_svg_path,
    width = clustered_size,
    height = clustered_size
  )
  draw_clustered_heatmap(matrix_values, matrix_label, lower_threshold, upper_threshold)
  grDevices::dev.off()

  grDevices::svg(
    simple_svg_path,
    width = simple_size,
    height = simple_size
  )
  draw_simple_heatmap(matrix_values, lower_threshold, upper_threshold)
  grDevices::dev.off()

  grDevices::png(
    clustered_png_path,
    width = clustered_size,
    height = clustered_size,
    units = "in",
    res = 150
  )
  draw_clustered_heatmap(matrix_values, matrix_label, lower_threshold, upper_threshold)
  grDevices::dev.off()

  grDevices::png(
    simple_png_path,
    width = simple_size,
    height = simple_size,
    units = "in",
    res = 150
  )
  draw_simple_heatmap(matrix_values, lower_threshold, upper_threshold)
  grDevices::dev.off()

  message(sprintf("Wrote %s", clustered_svg_path))
  message(sprintf("Wrote %s", simple_svg_path))
  message(sprintf("Wrote %s", clustered_png_path))
  message(sprintf("Wrote %s", simple_png_path))
}

main <- function() {
  # Run the FastAAI matrix visualisation workflow.
  parsed_args <- parse_args()
  matrix_values <- read_matrix_file(parsed_args$matrix_path)
  write_outputs(
    matrix_values,
    parsed_args$matrix_path,
    parsed_args$lower_threshold,
    parsed_args$upper_threshold
  )
  0
}

quit(save = "no", status = main())
