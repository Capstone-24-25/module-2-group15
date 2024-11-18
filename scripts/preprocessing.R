## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

nlp_fn_multi <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'mclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

nlp_fn_test <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, mclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

nlp_fn_bigrams <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = bigram, 
                  input = text_clean, 
                  token = 'ngrams', 
                  n = 2, 
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    separate(bigram, into = c("word1", "word2"), sep = " ") %>%
    unite("bigram", word1, word2, sep = " ") %>%
    count(.id, bclass, bigram, name = 'n') %>%
    bind_tf_idf(term = bigram, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'bigram',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# Projection functions
projection_fn <- function(.dtm, .prop){
  # coerce feature matrix to sparse
  dtm_mx <- .dtm %>%
    as.matrix() %>%
    as('sparseMatrix')
  
  # compute svd
  svd_out <- svd(dtm_mx) # svd function: singular value decomposition
  
  # select number of projections
  var_df <- tibble(var = svd_out$d^2) %>%
    mutate(pc = row_number(),
           cumulative = cumsum(var)/sum(var))
  
  n_pc <- which.min(var_df$cumulative < .prop)
  
  # extract loadings
  loadings <- svd_out$v[, 1:n_pc] %>% as.matrix()
  
  # extract scores
  scores <- (dtm_mx %*% svd_out$v[, 1:n_pc]) %>% as.matrix()
  
  # adjust names
  colnames(loadings) <- colnames(scores) <- paste('pc', 1:n_pc, sep = '')
  
  # output
  out <- list(n_pc = n_pc,
              var = var_df,
              projection = loadings,
              data = as_tibble(scores))
  
  return(out)
}

reproject_fn <- function(.dtm, .projection_fn_out){
  as_tibble(as.matrix(.dtm) %*% .projection_fn_out$projection)
}

# prelim1_data = parse_data(claims_raw)

# save(prelim1_data, file = 'data/prelim1_data.RData')