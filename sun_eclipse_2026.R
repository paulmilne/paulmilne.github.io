library(suncalc)
library(googleway)
library(dplyr)
library(lubridate)
library(ggplot2)
library(jpeg)
library(httr)

# 1. Define location and time sequence
# 52.893608 N, 1.265934 W (near Nottinghamshire/Lincolnshire border)
lat <- 52.893608
lon <- -1.265934
date <- as.Date("2026-08-12") # Partial solar eclipse over the UK

# Eclipse locally runs roughly 18:00-20:00 BST (17:00-19:00 UTC);
# widen slightly either side to give context on the sun's path
times <- seq(
  from = as.POSIXct(paste(date, "16:30:00"), tz = "UTC"),
  to   = as.POSIXct(paste(date, "19:15:00"), tz = "UTC"),
  by   = "1 min"
)

# 2. Calculate sun positions (Azimuth & Altitude)
sun_path <- getSunlightPosition(date = times, lat = lat, lon = lon) |>
  mutate(
    # Convert azimuth (radians south-to-west) to compass degrees (0-360° from North)
    azimuth_deg = (azimuth * 180 / pi + 180) %% 360,
    altitude_deg = altitude * 180 / pi,
    time_bst = format(with_tz(date, "Europe/London"), "%H:%M")
  )

# 3. Retrieve Google Street View aimed at the sun at maximum eclipse
# Reported UK-wide maximum eclipse is ~19:05-19:13 BST (18:05-18:13 UTC);
# pick the row closest to 19:09 BST (18:09 UTC) for this longitude
peak_time <- as.POSIXct(paste(date, "18:09:00"), tz = "UTC")
peak_sun <- sun_path[which.min(abs(sun_path$date - peak_time)), ]

api_key <- Sys.getenv("GOOGLE_MAPS_KEY")

size <- c(640, 640) # square image => assume equal horizontal/vertical FOV
fov <- 90            # Street View horizontal field of view (degrees)
heading <- peak_sun$azimuth_deg   # Camera pointed toward the sun at peak eclipse
pitch <- peak_sun$altitude_deg    # Camera tilted up to the sun's elevation

# googleway::google_streetview() only renders the image in the Viewer/HTML;
# to overlay the sun's path we need the raw pixels, so hit the Static Street
# View API directly and read the JPEG bytes into an array.
sv_url <- modify_url(
  "https://maps.googleapis.com/maps/api/streetview",
  query = list(
    location = paste(lat, lon, sep = ","),
    size     = paste(size, collapse = "x"),
    heading  = heading,
    fov      = fov,
    pitch    = pitch,
    key      = api_key
  )
)

sv_tmp <- tempfile(fileext = ".jpg")
sv_resp <- GET(sv_url, write_disk(sv_tmp, overwrite = TRUE))
stop_for_status(sv_resp)
streetview_img <- readJPEG(sv_tmp)

img_w <- size[1]
img_h <- size[2]

# 4. Project the sun's path onto the image plane.
# This assumes a simple linear (small-angle) mapping from angular offset to
# pixel offset, which is only a good approximation because the eclipse
# window spans a modest range of azimuth/altitude; it's not a true
# perspective/gnomonic projection.
wrap180 <- function(x) ((x + 180) %% 360) - 180

sun_projected <- sun_path |>
  mutate(
    delta_az = wrap180(azimuth_deg - heading),
    delta_alt = altitude_deg - pitch,
    x = img_w / 2 + (delta_az / (fov / 2)) * (img_w / 2),
    y = img_h / 2 + (delta_alt / (fov / 2)) * (img_h / 2)
  ) |>
  # Keep only points that actually fall within the camera's field of view
  filter(abs(delta_az) <= fov / 2, abs(delta_alt) <= fov / 2)

peak_projected <- sun_projected |> filter(date == peak_sun$date)

ggplot() +
  annotation_raster(
    streetview_img,
    xmin = 0, xmax = img_w, ymin = 0, ymax = img_h
  ) +
  geom_path(
    data = sun_projected,
    aes(x = x, y = y),
    color = "yellow", linewidth = 1
  ) +
  geom_point(
    data = peak_projected,
    aes(x = x, y = y),
    color = "red", size = 3
  ) +
  geom_text(
    data = peak_projected,
    aes(x = x, y = y, label = time_bst),
    color = "red", vjust = -1
  ) +
  coord_fixed(xlim = c(0, img_w), ylim = c(0, img_h), expand = FALSE) +
  labs(
    title = "Sun's path during the 12 Aug 2026 partial solar eclipse",
    subtitle = paste0("Camera heading ", round(heading, 1), "°, pitch ",
                       round(pitch, 1), "° — near Nottinghamshire/Lincolnshire border"),
    x = NULL, y = NULL
  ) +
  theme_void() +
  theme(
    plot.title = element_text(color = "white"),
    plot.subtitle = element_text(color = "white"),
    plot.background = element_rect(fill = "black")
  )
