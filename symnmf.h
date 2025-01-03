int allocate_matrix(double ***mat, int rows, int cols);

int free_matrix(double ***mat, int rows);

int calc_sym(double ***A, double ***X, int rows_x, int cols_x);

int calc_ddg(double ***D, double ***A, int n);

int calc_norm(double ***W, double ***D, double ***A, int n);

int calc_symnmf(double ***H_optimized, double ***H_initial, int h_rows, int h_cols, double ***W, int w_size);

void print_matrix(double **mat, int rows, int cols);