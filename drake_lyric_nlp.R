#!/usr/bin/env Rscript
# drake_lyric_nlp.R — Compliant rvest + LRCLIB Discography Sentiment & Lexical Pipeline (2009–2026)

suppressPackageStartupMessages({
  library(tidyverse)
  library(httr)
  library(jsonlite)
  library(rvest)
  library(tidytext)
  library(textdata)
  library(svglite)
  library(scales)
})

dir.create("data", showWarnings = FALSE, recursive = TRUE)
dir.create("outputs", showWarnings = FALSE, recursive = TRUE)

# 1. Verify Public Discography Metadata via rvest (Graceful Offline Fallback)
fetch_wiki_studio_years <- function() {
  url <- "https://en.wikipedia.org/wiki/Drake_albums_discography"
  tryCatch({
    resp <- httr::GET(url, httr::user_agent("DrakeLyricNLP/1.0 (Academic Research; NMAMIT CSE)"))
    if (httr::status_code(resp) != 200) return(tibble(album_key = character(), wiki_year = integer()))
    doc <- rvest::read_html(httr::content(resp, as = "text", encoding = "UTF-8"))
    tables <- rvest::html_elements(doc, "table.wikitable")
    if (length(tables) == 0) return(tibble(album_key = character(), wiki_year = integer()))
    raw_tbl <- rvest::html_table(tables[[1]], fill = TRUE)
    names(raw_tbl) <- make.unique(tolower(gsub("[^a-zA-Z0-9]+", "_", names(raw_tbl))))
    title_col <- names(raw_tbl)[1]
    details_col <- names(raw_tbl)[min(2, ncol(raw_tbl))]
    tibble(
      album_key = tolower(trimws(as.character(raw_tbl[[title_col]]))),
      wiki_year = suppressWarnings(as.integer(stringr::str_extract(as.character(raw_tbl[[details_col]]), "\\b20[0-2][0-9]\\b")))
    ) |> filter(!is.na(wiki_year)) |> distinct(album_key, .keep_all = TRUE)
  }, error = function(e) tibble(album_key = character(), wiki_year = integer()))
}

track_manifest <- tribble(
  ~album, ~year, ~cohort, ~track_name,
  "So Far Gone", 2009, "Heartbreak_RnB", "Best I Ever Had",
  "So Far Gone", 2009, "Heartbreak_RnB", "Houstatlantavegas",
  "So Far Gone", 2009, "Heartbreak_RnB", "Successful",
  "Thank Me Later", 2010, "Hybrid", "Fireworks",
  "Thank Me Later", 2010, "Hybrid", "Over",
  "Thank Me Later", 2010, "Hybrid", "Find Your Love",
  "Take Care", 2011, "Heartbreak_RnB", "Marvins Room",
  "Take Care", 2011, "Heartbreak_RnB", "Shot for Me",
  "Take Care", 2011, "Heartbreak_RnB", "Doing It Wrong",
  "Take Care", 2011, "Heartbreak_RnB", "Crew Love",
  "Nothing Was the Same", 2013, "Hybrid", "Tuscan Leather",
  "Nothing Was the Same", 2013, "Hybrid", "From Time",
  "Nothing Was the Same", 2013, "Hybrid", "Hold On We're Going Home",
  "If Youre Reading This Its Too Late", 2015, "Hype_Rap", "Energy",
  "If Youre Reading This Its Too Late", 2015, "Hype_Rap", "10 Bands",
  "If Youre Reading This Its Too Late", 2015, "Hype_Rap", "Know Yourself",
  "If Youre Reading This Its Too Late", 2015, "Hype_Rap", "6 God",
  "Views", 2016, "Heartbreak_RnB", "Keep the Family Close",
  "Views", 2016, "Heartbreak_RnB", "Redemption",
  "Views", 2016, "Heartbreak_RnB", "With You",
  "Views", 2016, "Heartbreak_RnB", "Faithful",
  "More Life", 2017, "Hybrid", "Passionfruit",
  "More Life", 2017, "Hybrid", "Gyalchester",
  "Scorpion", 2018, "Hybrid", "Nonstop",
  "Scorpion", 2018, "Hybrid", "Gods Plan",
  "Scorpion", 2018, "Hybrid", "Emotionless",
  "Certified Lover Boy", 2021, "Heartbreak_RnB", "Champagne Poetry",
  "Certified Lover Boy", 2021, "Heartbreak_RnB", "Pipe Down",
  "Certified Lover Boy", 2021, "Heartbreak_RnB", "Race My Mind",
  "Her Loss", 2022, "Hype_Rap", "Rich Flex",
  "Her Loss", 2022, "Hype_Rap", "Major Distribution",
  "Her Loss", 2022, "Hype_Rap", "On BS",
  "For All the Dogs", 2023, "Hybrid", "7969 Santa",
  "For All the Dogs", 2023, "Hybrid", "First Person Shooter",
  "For All the Dogs", 2023, "Hybrid", "Polar Opposites",
  "2025 Releases", 2025, "Hype_Rap", "What Did I Miss",
  "2026 Studio Drops", 2026, "Hybrid", "Solar Eclipse"
)

# 2. Legal Open-API Lyric Fetcher via LRCLIB (Polite Throttling + Local Cache)
fetch_lrclib_track <- function(track_name, album_name, artist = "Drake") {
  Sys.sleep(0.35)
  ua <- httr::add_headers("User-Agent" = "DrakeLyricNLP/1.0 (https://github.com/prajwal918/drake-lyric-nlp)")
  resp <- httr::GET("https://lrclib.net/api/get", query = list(artist_name = artist, track_name = track_name, album_name = album_name), ua)
  if (httr::status_code(resp) == 429) {
    wait_sec <- suppressWarnings(as.numeric(httr::headers(resp)[["retry-after"]]))
    Sys.sleep(if (is.na(wait_sec)) 5 else wait_sec)
    resp <- httr::GET("https://lrclib.net/api/get", query = list(artist_name = artist, track_name = track_name), ua)
  }
  if (httr::status_code(resp) != 200) {
    search_resp <- httr::GET("https://lrclib.net/api/search", query = list(artist_name = artist, track_name = track_name), ua)
    if (httr::status_code(search_resp) != 200) return(NA_character_)
    items <- jsonlite::fromJSON(httr::content(search_resp, as = "text", encoding = "UTF-8"))
    if (!is.data.frame(items) || nrow(items) == 0 || !"plainLyrics" %in% names(items)) return(NA_character_)
    return(as.character(items$plainLyrics[[1]]))
  }
  parsed <- jsonlite::fromJSON(httr::content(resp, as = "text", encoding = "UTF-8"))
  as.character(parsed$plainLyrics)
}

# Moving-Average Type-Token Ratio (MATTR, window = 50)
compute_mattr <- function(words, window = 50L) {
  n <- length(words)
  if (n <= window) return(n_distinct(words) / max(1L, n))
  mean(vapply(seq_len(n - window + 1L), function(i) {
    n_distinct(words[i:(i + window - 1L)]) / window
  }, numeric(1)))
}

cat("Drake Lyric NLP Pipeline initialized successfully.\n")
