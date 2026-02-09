# MATRIX SIMILARITY (Human vs Synthetic Pop/Persona)

# Data loading
human   <- read.csv("human_traits.csv")
pop     <- read.csv("synthetic_population_traits.csv")
persona <- read.csv("synthetic_persona_traits.csv")

traits <- c("CON","AGR","EXT","NEU","OPE")

# Simple helper to compute correlation matrix (pairwise complete)
cor_mat <- function(df) {
  cor(df[, traits], use = "pairwise.complete.obs", method = "pearson")
}

# Method for extracting the 10 unique off-diagonal correlations (upper triangle)
vec_upper <- function(M) {
  M[upper.tri(M, diag = FALSE)]
}

# Compute matrices
R_h <- cor_mat(human)
R_p <- cor_mat(pop)
R_s <- cor_mat(persona)

# Vectorize
v_h <- vec_upper(R_h)
v_p <- vec_upper(R_p)
v_s <- vec_upper(R_s)

# Similarity indices: correlation between vectorized upper triangles
sim_h_pop     <- cor(v_h, v_p, method = "pearson")
sim_h_persona <- cor(v_h, v_s, method = "pearson")

# Print results
cat("Matrix similarity (Human vs Synthetic Population):", round(sim_h_pop, 3), "\n")
cat("Matrix similarity (Human vs Synthetic Persona):", round(sim_h_persona, 3), "\n")
