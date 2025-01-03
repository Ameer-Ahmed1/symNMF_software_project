#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_ITERATION 300
#define BETA 0.5
#define EPSILON 0.0001

/*
    * handle errors as instructed
*/
void handle_error() {
    printf("An Error Has Occurred");
}

/*
    * Calculate the squared euclidean distance
    * return dist
    * 
    * *x_i: a vector of n data points
    * *x_j: a vector of n data points
*/
double calc_sqr_euclidean_dist(double *x_i, double *x_j, int n) {
    double dist = 0;
    double diff;
    int i;

    for(i = 0; i < n; i++) {
        diff = x_i[i] - x_j[i];
        dist += pow(diff, 2);
    }

    return dist;
}

/*
    * free a matrix from memory
    * returns 0 on success
    * 
    * mat: a matrix
    * rows: number of rows in mat
*/
int free_matrix(double ***mat, int rows) {
    int i;
    
    /* Free each row */
    for (i = 0; i < rows; i++) { 
        free((*mat)[i]);
    }

    free((*mat)); /* Free the array of row pointers */ 
    return 0; /* Success */ 
}


/*
    * Allocate memory for the given *mat
    * pre: mat is not dynamically allocated
    * return 0 on success
    * 
    * rows: the number of rows to allocate
    * cols: the number of cols to allocate
*/
int allocate_matrix(double ***mat, int rows, int cols) {
    int i;

    /* Allocate memory for the array of pointers to rows */ 
    *mat = (double**)malloc(rows * sizeof(double *)); 
    if (*mat == NULL) {
        return 1; /* Memory allocation failed */
    }

    /* Allocate memory for each row */
    for (i = 0; i < rows; i++) { 
        (*mat)[i] = (double *)calloc(cols, sizeof(double));

        if ((*mat)[i] == NULL) {
            free_matrix(mat, i);
            return 1; /* Memory allocation failed */ 
        }
    }

    return 0; /* Success */  
}

/*
    * calculate the diagonal matrix *D risen to the power of p and store in * result
    * pre: *result is not dynamically allocated
    * return 0 on success
    * 
    * n: number of rows and column in D and result
    * p: the power
*/
int pow_ddg_matrix(double ***result, double ***D, int n, double p) {
    int i;
    int j;

    if(allocate_matrix(result, n, n) != 0) {
        return 1;
    }

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++)
        if(i == j) {
            (*result)[i][j] = pow((*D)[i][i], p);
        } else {
            (*result)[i][j] = 0;
        }
        
    }


    return 0;
}

/*
    * calculate the product of matrix D with matrix A
    * pre: result is not dynamically allocated
    * pre: assuming D is diagonal and A is the similarity matrix
    * return 0 on success
    *
    * *result: the matrix in which the product is saved
    * *D: n*n diagonal matrix
    * *A: n*n similarity matrix
    * ddg_on_left: 1 if the diagonal matrix is on left side 0 if its on right 
*/
int mul_ddg_mat(double ***result, double ***D, double ***A, int n, int ddg_on_left) {
    int i,j;

    if(allocate_matrix(result, n, n) != 0) {
        return 1;
    }

    for(i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {

            if(ddg_on_left == 1) {
                (*result)[i][j] = (*D)[i][i] * (*A)[i][j];

            } else if(ddg_on_left == 0) {
                (*result)[i][j] = (*D)[j][j] * (*A)[i][j];
            }
            
        }
    }

    return 0;
}

/*
    * Calculate the transpose of a given matrix
    * pre: result is not dynamically allocated
    * return 0 on success
    * 
    * result: A pointer to a matrix of cols * rows entries
    * mat: A pointer to a matrix of rows * cols entries
*/
int transpose(double ***result, double ***mat, int rows, int cols) {
    int i;
    int j;

    if(allocate_matrix(result, cols, rows) != 0) {
        handle_error();
        return 1;
    }

    for(i = 0; i < cols; i++) {
        for(j = 0; j < rows; j++) {
            (*result)[i][j] = (*mat)[j][i];
        }
    }
    return 0;
}

/*
    * calculate the matrix A - B
    * pre: result is dynamically allocated
    * return 0 on sucess
    * 
    * result: Pointer to a matrix of rows * cols entries
    * A: Pointer to a matrix of rows * cols entries
    * B: Pointer to a matrix of rows * cols entries
*/
int subtract_matrices(double ***result, double ***A, double ***B, int rows, int cols) {
    int i;
    int j;

    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            (*result)[i][j] = (*A)[i][j] - (*B)[i][j];
        }
    }

    return 0;
}

/*
    * Calculate the Frobenius norm squared of a matrix A
    * return ΣΣ(a_ij)^2 on success
    * 
    * A: Pointer to a matrix of rows * cols entries
*/
double frobenius_norm_squared(double ***A, int rows, int cols) {
    double f_norm_squared = 0;
    int i, j;
    
    for(i = 0; i < rows; i++) {
        for(j = 0; j < cols; j++) {
            f_norm_squared += pow((*A)[i][j], 2);
        }
    }

    return f_norm_squared;
}

/*
    * Calculate the matrix A * B
    * pre: result is not dynamically allocated
    * assume: cols_A = rows_B
    * return 0 on success
    * 
    * result: Pointer to a matrix of rows_A * cols_B entries
    * A: Pointer to a matrix of rows_A * cols_A entries
    * B: Pointer to a matrix of rows_B * cols_B entries
*/
int multiply_matrices(double ***result, double ***A, int rows_A, int cols_A, double ***B, int cols_B) {
    int i, j, k;

    if(allocate_matrix(result, rows_A, cols_B) != 0 ) {
        handle_error();
        return 1;
    }

    for (i = 0; i < rows_A; i++) {
        for (j = 0; j < cols_B; j++) {
            (*result)[i][j] = 0;
            for (k = 0; k < cols_A; k++) {
                (*result)[i][j] += (*A)[i][k] * (*B)[k][j];
            }
        }
    }
    return 0; 
}

/*
    *calculate H_updated given H_in
    *pre: h_updated is dynamically allocated
    * return 0 on success
    * 
    * H_updated: Pointer to a matrix of h_rows * h_cols entries
    * H_in: Pointer to the given matrix to update of h_rows * h_cols entries
    * W: Pointer to the normalized similarity matrix of w_size * w_size entries
*/
int update_H(double ***H_updated, double***H_in, int h_rows, int h_cols, double ***W, int w_size) {
    double **H_transpose; /* H_in transposed */ 
    double **H_Ht; /* H_in * H_transpose */ 
    double **H_Ht_H; /* H_in * H_transpose * H_in */ 
    double **W_H; /* W * H_in */ 
    int i, j;

    if(transpose(&H_transpose, H_in, h_rows, h_cols) != 0) {
        free_matrix(&H_transpose, h_cols);
        handle_error();
        return 1;
    }

    if(multiply_matrices(&H_Ht, H_in, h_rows, h_cols, &H_transpose, h_rows) != 0) {
        free_matrix(&H_transpose, h_cols);
        free_matrix(&H_Ht, h_rows);
        handle_error();
        return 1;
    }

    if(multiply_matrices(&H_Ht_H, &H_Ht, h_rows, h_rows, H_in, h_cols) != 0) {
        free_matrix(&H_Ht_H, h_rows);
        free_matrix(&H_transpose, h_cols);
        free_matrix(&H_Ht, h_rows);
        handle_error();
        return 1;
    }

    if(multiply_matrices(&W_H, W, w_size, w_size, H_in, h_cols) != 0) {
        free_matrix(&W_H, w_size);
        free_matrix(&H_transpose, h_cols);
        free_matrix(&H_Ht, h_rows);
        free_matrix(&H_Ht_H, h_rows);
        handle_error();
        return 1;
    }

    for(i = 0; i < h_rows; i ++) {
        for(j = 0; j < h_cols; j ++) {

            if(H_Ht_H[i][j] == 0) { /* division by zero is not allowed*/
                free_matrix(&W_H, w_size);
                free_matrix(&H_transpose, h_cols);
                free_matrix(&H_Ht, h_rows);
                free_matrix(&H_Ht_H, h_rows);
                handle_error();
                return 1;
            }

            (*H_updated)[i][j] = (*H_in)[i][j] * (1 - BETA + BETA * (W_H[i][j] / H_Ht_H[i][j])); /* update H[i,j] as instructed */ 
        }
    }

    free_matrix(&H_transpose, h_cols);
    free_matrix(&H_Ht, h_rows);
    free_matrix(&H_Ht_H, h_rows);
    free_matrix(&W_H, w_size);
    
    return 0;
}

/*
    * Calculate the similarity matrix *A 
    * pre: *A is not dynamically allocated
    * return 0 on success
    * 
    * *X: a matrix of rows_x data points where each consists of cols_x values
*/
int calc_sym(double ***A, double ***X, int rows_x, int cols_x) {
    int i,j;
    double a_ij;

    if(allocate_matrix(A, rows_x, rows_x) != 0) {
        handle_error();
        return 1;
    }

    for(i = 0; i < rows_x; i++) {
        (*A)[i][i] = 0;
        for(j = i+1; j < rows_x; j++) {
            a_ij = exp(calc_sqr_euclidean_dist((*X)[i], (*X)[j], cols_x) * -0.5);
            (*A)[i][j] = a_ij;
            (*A)[j][i] = a_ij;
        }
    }

    return 0;
}

/*
    * calculate the diagonal degree matrix given matrix A
    * pre: *D is not dynamically allocated
    * return 0 on success
    * 
    * *D: the ddg matrix
    * *A: the similarity matrix
    *  n: number of rows and columns in A and D
*/
int calc_ddg(double ***D, double ***A, int n) {
    int i,j;

    if(allocate_matrix(D, n, n) != 0) {
        handle_error();
        return 1;
    }

    for(i = 0; i < n; i++) {
        (*D)[i][i] = 0;
        for(j = 0; j < n; j++) {
            (*D)[i][i] += (*A)[i][j];
            if(i != j) {
                (*D)[i][j] = 0;
            }
        }
    } 

    return 0;
}

/*
    * calculate the normalized similarity matrix
    * pre: W is not dynamically allocated
    * return 0 on success
    * 
    * D: a diagonal matrix
    * A: a similarity matrix
    * n: number of rows and columns
*/
int calc_norm(double ***W, double ***D, double ***A, int n) {
    double **D_powered = NULL; /* D^(-0.5) */ 
    double **temp_res = NULL;

    if(pow_ddg_matrix(&D_powered, D, n, -0.5) != 0) {
        free_matrix(&temp_res, n);
        handle_error();
        return 1;
    }

    if(mul_ddg_mat(&temp_res, &D_powered, A, n, 1) !=0 ) {
        free_matrix(&D_powered, n);
        handle_error();
        return 1;
    }

    if(mul_ddg_mat(W, &D_powered, &temp_res, n, 0) != 0) {
        free_matrix(&temp_res, n);
        free_matrix(&D_powered, n);
        handle_error();
        return 1;
    }
    
    /* free allocated matrices */ 
    free_matrix(&D_powered, n);
    free_matrix(&temp_res, n);

    return 0;
}

/*
    * Calculate the optimized H
    * pre: H_optimized is not dynamically allocated
    * return 0 on success
    * 
    * H_optimized: Pointer to a matrix of h_rows * h_cols entries
    * H_initial: Pointer to a matrix of h_rows * h_cols entries
    * W: Pointer to a matrix of w_size * w_size entries (The normalized similarity matrix)
*/
int calc_symnmf(double ***H_optimized, double ***H_initial, int h_rows, int h_cols, double ***W, int w_size) {
    double f_norm_squared = EPSILON + 1;
    int iter = 0;
    int i, j;

    double **Diff; /* H_optimized - H_initial */ 

    if(allocate_matrix(H_optimized, h_rows, h_cols) != 0) {
        handle_error();
        return 1;
    }

    if(allocate_matrix(&Diff, h_rows, h_cols) != 0) {
        free_matrix(H_optimized, h_rows);
        handle_error();
        return 1;
    }
    
    while(iter < MAX_ITERATION && f_norm_squared >= EPSILON) {
        if(update_H(H_optimized, H_initial, h_rows, h_cols, W, w_size) != 0) {
            free_matrix(H_optimized, h_rows);
            free_matrix(&Diff, h_rows);
            handle_error();
            return 1;   
        }

        subtract_matrices(&Diff, H_optimized, H_initial, h_rows, h_cols);
        f_norm_squared = frobenius_norm_squared(&Diff, h_rows, h_cols);

        /* copy matrix H_optimized to H_initial */ 
        for(i = 0; i < h_rows; i++) {
            for(j = 0; j < h_cols; j++) {
                (*H_initial)[i][j] = (*H_optimized)[i][j];
            }
        }

        iter++;
    }

    free_matrix(&Diff, h_rows);
    return 0;    
}

/*
    * calculate the number of rows in file and columns
    * return 0 on success
    * 
    * filename: a pointer to the filename
    * rows: an address of int that points to the number of rows
    * cols: an address of int that points to the number of columns
*/
int get_file_size(const char *filename, int *rows, int *cols) {
    FILE *fp;
    char ch;

    fp = fopen(filename, "r");
    if(fp == NULL) {
        handle_error();
        return 1;
    }

    (*rows) = 0;
    (*cols) = 1;

    while ((ch = fgetc(fp)) != EOF) {
        if((*rows) == 0 && ch == ',') {
            (*cols)++;
        }
        if(ch == '\n') {
            (*rows)++;
        }
    }
    fclose(fp);

    return 0;
}

/*
    * given the goal check if its allowed
    * return 0 if the goal is allowed, 1 else
    * 
    * goal: a string that stores the goal of the user
*/
int is_goal_allowed(const char *goal) {
    int i;
    char *goals_allowed [] = {"sym", "ddg", "norm"};

    for(i = 0; i < 3; i++) {
        if(strcmp(goals_allowed[i], goal) == 0) {
            return 0;
        }
    }

    return 1;
}

/*
    * given filename and dimensions parse the data and store it in a matrix X
    * pre: X is not dynamically allocated
    * return 0 on success
    * 
    * X: a matrix to save the datapoints
    * filename: the filename from which we read data
    * rows: number of rows in filename
    * cols: number of cols in filename
*/
int store_data(double ***X, const char *filename, int *rows, int *cols) {
    FILE *fp;
    int n, m;
    char c;

    if (get_file_size(filename, rows, cols) != 0) {
        return 1;
    }

    if (allocate_matrix(X, *rows, *cols) != 0) {
        return 1;
    }

    fp = fopen(filename, "r");
    if (fp == NULL) {
        free_matrix(X, *rows);
        return 1;
    }

    n = 0;
    m = 0;

    while (fscanf(fp, "%lf", &(*X)[n][m]) != EOF) {
        m++;
        c = fgetc(fp);
        if (c == '\n' || c == EOF) {
            n++;
            m = 0;
        } else if (c != ',') {
            fclose(fp);
            free_matrix(X, *rows);
            return 1;
        }
        if (n >= *rows || m >= *cols) {
            break; 
        }
    }

    fclose(fp);

    return 0;
}

/*
    * given a matrix print its rows on separated lines with delimiter between each entry
    * 
    * mat: a matrix to print
    * rows: number of rows in matrix
    * cols: number of columns in matrix
*/
void print_matrix(double **mat, int rows, int cols) {
    int i,j;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            printf("%.4f", mat[i][j]);
            if (j < cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

int main(int argc, char const *argv[])
{
    double **data;
    const char *goal = argv[1];
    const char *file_name = argv[2];
    char *dot = strrchr(file_name, '.');
    int rows, cols;
    double **A;
    double **D;
    double **W;

    if(argc < 3) {
        handle_error();
        exit(1);
    }


    if(is_goal_allowed(goal) == 1) {
        handle_error();
        exit(1);
    }
    
    if (dot && strcmp(dot, ".txt")) {
        handle_error();
        exit(1);
    }

    if(store_data(&data, file_name, &rows, &cols) != 0) {
        handle_error();
        exit(1);
    }
    
    if(strcmp(goal, "sym") == 0) {
        calc_sym(&A, &data, rows, cols);
        print_matrix(A, rows, rows);
        free_matrix(&A, rows);
    }

    else if(strcmp(goal, "ddg") == 0) {
        calc_sym(&A, &data, rows, cols);
        calc_ddg(&D,&A, rows);
        print_matrix(D,rows,rows);
        free_matrix(&D, rows);
        free_matrix(&A, rows);
    }

    else if(strcmp(goal, "norm") == 0) {
        calc_sym(&A, &data, rows, cols);
        calc_ddg(&D, &A, rows);
        calc_norm(&W, &D, &A, rows);
        print_matrix(W, rows, rows);
        free_matrix(&W, rows);
        free_matrix(&D, rows);
        free_matrix(&A, rows);
    } else {
        handle_error();
        free_matrix(&data, rows);
        return 1;
    }

    free_matrix(&data, rows);

    return 0;
}
    
