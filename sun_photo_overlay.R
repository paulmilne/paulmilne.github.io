library(exiftoolr)
library(suncalc)
library(dplyr)
library(lubridate)
library(ggplot2)
library(jpeg)

photo_path <- "/Users/paul/Downloads/TH000003.JPG"

# 1. Read camera position/orientation and capture time from EXIF/XMP metadata.
# This photo was taken with the Theodolite app (iPhone), which records:
#   - GPS position + GPSImgDirection (true compass heading of the camera)
#   - vert_angle_deg / horiz_angle_deg in the ImageDescription/caption
#     (camera pitch and roll respectively)
meta <- exif_read(photo_path)

photo_lat <- meta$GPSLatitude
photo_lon <- meta$GPSLongitude
heading   <- meta$GPSImgDirection
pitch     <- as.numeric(sub(".*vert_angle_deg=([0-9.\\-]+).*", "\\1", meta$ImageDescription))
roll      <- as.numeric(sub(".*horiz_angle_deg=([0-9.\\-]+).*", "\\1", meta$ImageDescription))
# NOTE: assuming vert_angle_deg = pitch (up/down) and treating roll as
# negligible (it was 0.4 degrees for this photo) - not corrected for below.

photo_time_local <- with_tz(
  ymd_hms(meta$DateTimeOriginal, tz = "UTC") + hours(0), # placeholder, replaced below
  "UTC"
)
# DateTimeOriginal has no offset attached in EXIF; combine with OffsetTimeOriginal
photo_time_local <- as.POSIXct(
  paste(gsub(":", "-", substr(meta$DateTimeOriginal, 1, 10)), substr(meta$DateTimeOriginal, 12, 19)),
  tz = "UTC"
) - hours(as.numeric(substr(meta$OffsetTimeOriginal, 1, 3)))

img_w <- meta$ImageWidth
img_h <- meta$ImageHeight

# 2. Estimate the photo's actual field of view.
# FocalLengthIn35mmFormat gives a horizontal FOV under the standard 36mm-width
# convention; the saved image is a crop of the native ultra-wide sensor
# (assumed 4032x3024 for the iPhone 13 Pro Max ultra-wide camera), so we
# rescale using a rectilinear (tangent-plane) projection to the actual
# output pixel dimensions. This is an approximation - the true optical FOV
# of the ultra-wide lens may differ from this 35mm-equivalent estimate.
native_w <- 4032
f35 <- meta$FocalLengthIn35mmFormat
fov_h_native <- 2 * atan(18 / f35) * 180 / pi
fov_h <- 2 * atan(tan(fov_h_native / 2 * pi / 180) * (img_w / native_w)) * 180 / pi
fov_v <- 2 * atan((img_h / img_w) * tan(fov_h / 2 * pi / 180)) * 180 / pi

# 3. Sun path across the eclipse window, computed at the photo's location.
date <- as.Date("2026-08-12")
times <- seq(
  from = as.POSIXct(paste(date, "16:30:00"), tz = "UTC"),
  to   = as.POSIXct(paste(date, "19:15:00"), tz = "UTC"),
  by   = "1 min"
)
sun_path <- getSunlightPosition(date = times, lat = photo_lat, lon = photo_lon) |>
  mutate(
    azimuth_deg = (azimuth * 180 / pi + 180) %% 360,
    altitude_deg = altitude * 180 / pi,
    time_bst = format(with_tz(date, "Europe/London"), "%H:%M")
  )

# Sun position at the moment this specific photo was taken
capture_sun <- getSunlightPosition(date = photo_time_local, lat = photo_lat, lon = photo_lon) |>
  mutate(
    azimuth_deg = (azimuth * 180 / pi + 180) %% 360,
    altitude_deg = altitude * 180 / pi
  )

# 4. Project onto the photo's pixel coordinates (small-angle/linear
# approximation around the camera's heading/pitch, as with the Street View
# overlay - not a true perspective projection).
wrap180 <- function(x) ((x + 180) %% 360) - 180

project <- function(az, alt) {
  delta_az <- wrap180(az - heading)
  delta_alt <- alt - pitch
  data.frame(
    x = img_w / 2 + (delta_az / (fov_h / 2)) * (img_w / 2),
    y = img_h / 2 + (delta_alt / (fov_v / 2)) * (img_h / 2),
    delta_az = delta_az,
    delta_alt = delta_alt
  )
}

sun_projected <- cbind(sun_path, project(sun_path$azimuth_deg, sun_path$altitude_deg))
capture_projected <- cbind(capture_sun, project(capture_sun$azimuth_deg, capture_sun$altitude_deg))

in_frame <- function(x, y) x >= 0 & x <= img_w & y >= 0 & y <= img_h

path_in_frame <- sun_projected |> filter(in_frame(x, y))
capture_in_frame <- in_frame(capture_projected$x[1], capture_projected$y[1])

photo_img <- readJPEG(photo_path)

p <- ggplot() +
  annotation_raster(photo_img, xmin = 0, xmax = img_w, ymin = 0, ymax = img_h) +
  coord_fixed(xlim = c(0, img_w), ylim = c(0, img_h), expand = FALSE) +
  theme_void() +
  labs(
    title = "Eclipse sun path projected onto photo",
    subtitle = paste0(
      "Heading ", round(heading, 1), "°, pitch ", round(pitch, 1),
      "°, est. FOV ", round(fov_h, 0), "°x", round(fov_v, 0),
      "°, captured ", format(photo_time_local, "%H:%M UTC")
    ),
    x = NULL, y = NULL
  ) +
  theme(
    plot.title = element_text(color = "white", size = 11),
    plot.subtitle = element_text(color = "white", size = 9),
    plot.background = element_rect(fill = "black")
  )

if (nrow(path_in_frame) > 0) {
  p <- p +
    geom_path(data = path_in_frame, aes(x = x, y = y), color = "yellow", linewidth = 1)

  # Label every 15 minutes along the visible portion of the path
  path_ticks <- path_in_frame |> filter(minute(date) %% 15 == 0)
  p <- p +
    geom_point(data = path_ticks, aes(x = x, y = y), color = "yellow", size = 2) +
    geom_label(
      data = path_ticks, aes(x = x, y = y, label = time_bst),
      color = "black", fill = "yellow", alpha = 0.8, size = 2.8,
      linewidth = 0, label.padding = unit(0.12, "lines"), vjust = -0.5
    )
}

if (isTRUE(capture_in_frame)) {
  p <- p +
    geom_point(data = capture_projected, aes(x = x, y = y), color = "red", size = 3) +
    geom_label(data = capture_projected, aes(x = x, y = y, label = "sun at capture"),
               color = "red", fill = "white", alpha = 0.8, linewidth = 0, vjust = -0.7)
} else {
  # Sun (at capture time) is off-frame: draw a clamped arrow pointing toward it
  # and annotate how far off it is, in degrees.
  cx <- pmin(pmax(capture_projected$x, 0), img_w)
  cy <- pmin(pmax(capture_projected$y, 0), img_h)
  offframe_label <- sprintf(
    "Sun at capture (%s) is %.0f° off-frame\n(Az %.0f°, Alt %.0f° vs heading %.0f°, pitch %.0f°)",
    format(photo_time_local, "%H:%M UTC"),
    sqrt(capture_projected$delta_az^2 + capture_projected$delta_alt^2),
    capture_sun$azimuth_deg, capture_sun$altitude_deg, heading, pitch
  )
  p <- p +
    annotate("segment", x = img_w / 2, y = img_h / 2, xend = cx, yend = cy,
             arrow = arrow(length = unit(0.3, "cm")), color = "red", linewidth = 1) +
    annotate("label", x = img_w * 0.02, y = img_h * 0.02, label = offframe_label,
             color = "red", fill = "white", alpha = 0.85, size = 3, hjust = 0, vjust = 0,
             linewidth = 0)
}

if (nrow(path_in_frame) == 0) {
  p <- p +
    annotate("label", x = img_w * 0.02, y = img_h * 0.98,
             label = "Eclipse sun path (16:30-19:15 UTC) never enters this frame",
             color = "black", fill = "yellow", alpha = 0.85, size = 3, hjust = 0, vjust = 1,
             linewidth = 0)
}

p
