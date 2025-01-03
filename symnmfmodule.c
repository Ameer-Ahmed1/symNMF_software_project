#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#include "symnmf.h"

/*
    * parse a python list of lists to a C matrix
    * pre: c_matrix is not dynamically allocated
    * return 0 on success
    * 
    * py_matrix: pointer to a 2d list with rows * cols dimensions
*/
int convert_python_PyObject_to_C_matrix(PyObject **py_matrix, double ***c_matrix, int rows, int cols) {
    int i, j;
    PyObject *row, *item;

    if (allocate_matrix(c_matrix, rows, cols) != 0) {
        return 1;  
    }

    for (i = 0; i < rows; i++) {
        row = PyList_GetItem(*py_matrix, i);
        if (row == NULL) {
            free_matrix(c_matrix, rows);
            return 1;  
        }
        for (j = 0; j < cols; j++) {
            item = PyList_GetItem(row, j);
            if (item == NULL) {
                free_matrix(c_matrix, rows);
                return 1;  
            }
            (*c_matrix)[i][j] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                free_matrix(c_matrix, rows);
                return 1;  
            }
        }
    }
    return 0;
}

/*  
    * convert a C matric to python list of lists
    * return the paresed python list on success
    * 
    * c_matrix: C 2D array of size rows * cols
*/
int convert_C_matrix_to_python_PyObject(PyObject **py_matrix, double ***c_matrix, int rows, int cols) { 
    
    int i, j;
    PyObject *row, *val;

    *py_matrix = PyList_New(rows);
    if (*py_matrix == NULL) {
        return 1;  
    }

    for (i = 0; i < rows; i++) {
        row = PyList_New(cols);
        if (row == NULL) {
            Py_DECREF(*py_matrix);
            return 1;  
        }

        for (j = 0; j < cols; j++) {
            val = PyFloat_FromDouble((*c_matrix)[i][j]);
            if (val == NULL) {
                Py_DECREF(row);
                Py_DECREF(*py_matrix);
                return 1;  
            }
            PyList_SetItem(row, j, val);
        }
        PyList_SetItem(*py_matrix, i, row);
    }
    return 0;
}

// return the similarity matrix on success, NULL on failure
static PyObject* sym(PyObject* self, PyObject* args) {
    PyObject *X_py;
    PyObject *A_py;
    double **X, **A;
    int rows_x, cols_x;
    
    if (!PyArg_ParseTuple(args, "Oii", &X_py, &rows_x, &cols_x)) {
        return NULL;
    }

    // Convert PyObject to C array
    if(convert_python_PyObject_to_C_matrix(&X_py, &X, rows_x, cols_x) != 0) {
        return NULL;
    }


    if (calc_sym(&A, &X, rows_x, cols_x) != 0) {
        free_matrix(&X, rows_x);
        return NULL;
    }

    // Convert C array A to PyObject
    convert_C_matrix_to_python_PyObject(&A_py, &A, rows_x, rows_x);

    free_matrix(&X, rows_x);
    free_matrix(&A, rows_x);

    return A_py;
}

// return the diagonal degree matrix on success, NULL on failure
static PyObject* ddg(PyObject* self, PyObject* args) {
    PyObject *X_py;
    PyObject *D_py;
    double **X, **A, **D;
    int rows, cols;

    if (!PyArg_ParseTuple(args, "Oii", &X_py, &rows, &cols)) {
        return NULL;
    }

    if(convert_python_PyObject_to_C_matrix(&X_py, &X, rows, cols) != 0) {
        return NULL;
    }

    if(calc_sym(&A, &X, rows, cols) != 0) {
        free_matrix(&X, rows);
        return NULL;
    }

    if(calc_ddg(&D, &A, rows) != 0) {
        free_matrix(&A, rows);
        free_matrix(&X, rows);
        return NULL;
    }

     if (convert_C_matrix_to_python_PyObject(&D_py, &D, rows, rows) != 0) {
        free_matrix(&A, rows);
        free_matrix(&X, rows);
        free_matrix(&D, rows);
        return NULL;
    }

    free_matrix(&A, rows);
    free_matrix(&D, rows);
    free_matrix(&X, rows);

    return D_py;
}

static PyObject* norm(PyObject* self, PyObject* args) {
    PyObject *X_py;
    PyObject *W_py;
    double **X, **A, **D, **W;
    int rows, cols;

    if (!PyArg_ParseTuple(args, "Oii", &X_py, &rows, &cols)) {
        return NULL;
    }
    

    if(convert_python_PyObject_to_C_matrix(&X_py, &X, rows, cols) != 0) {
        return NULL;
    }

    if(calc_sym(&A, &X, rows, cols) != 0) {
        free_matrix(&X, rows);
        return NULL;
    }

    if(calc_ddg(&D, &A, rows) != 0) {
        free_matrix(&A, rows);
        free_matrix(&X, rows);
        return NULL;
    }

    if(calc_norm(&W, &D, &A, rows) != 0) {
        free_matrix(&A, rows);
        free_matrix(&X, rows);
        free_matrix(&D, rows);
        return NULL;
    }

    convert_C_matrix_to_python_PyObject(&W_py, &W, rows, rows);

    free_matrix(&X, rows);
    free_matrix(&A, rows);
    free_matrix(&D, rows);
    
    return W_py;
}

// return the optimal H matrix on success, NULL on failure
static PyObject* symnmf(PyObject* self, PyObject* args) {
    PyObject *W_py;
    PyObject *H_optimized_py;
    PyObject *H_init_py;
    double **H_init, **H_optimized, **W;
    int rows_h, cols_h, size_w;

    if (!PyArg_ParseTuple(args, "OO", &H_init_py, &W_py)) {
        return NULL;
    }
    rows_h = PyList_Size(H_init_py);
    cols_h = PyList_Size(PyList_GetItem(H_init_py, 0));
    size_w = PyList_Size(W_py);

    if(convert_python_PyObject_to_C_matrix(&H_init_py, &H_init, rows_h, cols_h) != 0) {
        free_matrix(&H_init, rows_h);
        return NULL;
    }

    if(convert_python_PyObject_to_C_matrix(&W_py, &W, size_w, size_w) != 0) {
        free_matrix(&H_init, rows_h);
        return NULL;
    }


    if(calc_symnmf(&H_optimized, &H_init, rows_h, cols_h, &W, size_w) != 0) {
        free_matrix(&H_init, rows_h);
        free_matrix(&W, size_w);
        free_matrix(&H_optimized, rows_h);
        return NULL;
    }

    convert_C_matrix_to_python_PyObject(&H_optimized_py, &H_optimized, rows_h, cols_h);

    free_matrix(&H_init, rows_h);
    free_matrix(&W, size_w);
    free_matrix(&H_optimized, rows_h);

    return H_optimized_py;
}


// module's function table
static PyMethodDef symnmf_functions_table[] = {
    {
        "sym", 
        (PyCFunction)sym, 
        METH_VARARGS, 
        "return the similarity matrix" 
    },  {
        "ddg", 
        (PyCFunction)ddg, 
        METH_VARARGS, 
        "return the diagonal degree matrix" 
    },  {
        "norm", 
        (PyCFunction)norm, 
        METH_VARARGS, 
        "return the normalized similarity matrix" 
    },  {
        "symnmf", 
        (PyCFunction)symnmf, 
        METH_VARARGS, 
        "return H optimized matrix" 
    },{
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule",     
    "Pyton wrapper for symnmf C functions.", 
    -1,
    symnmf_functions_table
};

PyMODINIT_FUNC PyInit_symnmfmodule(void) {
    return PyModule_Create(&symnmfmodule);
}