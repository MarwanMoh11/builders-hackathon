#define PY_SSIZE_T_CLEAN
#include <Python.h>

static inline PyObject *concat_tuples(PyObject *a, PyObject *b) {
    if (PyTuple_CheckExact(a) && PyTuple_CheckExact(b)) {
        Py_ssize_t na = PyTuple_GET_SIZE(a);
        Py_ssize_t nb = PyTuple_GET_SIZE(b);
        PyObject *out = PyTuple_New(na + nb);
        if (!out) return NULL;
        for (Py_ssize_t i = 0; i < na; i++) {
            PyObject *item = PyTuple_GET_ITEM(a, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(out, i, item);
        }
        for (Py_ssize_t i = 0; i < nb; i++) {
            PyObject *item = PyTuple_GET_ITEM(b, i);
            Py_INCREF(item);
            PyTuple_SET_ITEM(out, na + i, item);
        }
        return out;
    }
    return PyNumber_Add(a, b);
}

static inline PyObject *get_row_item(PyObject *row, Py_ssize_t idx) {
    if (PyTuple_CheckExact(row)) {
        if (idx >= 0 && idx < PyTuple_GET_SIZE(row)) {
            PyObject *v = PyTuple_GET_ITEM(row, idx);
            Py_INCREF(v);
            return v;
        }
    } else if (PyList_CheckExact(row)) {
        if (idx >= 0 && idx < PyList_GET_SIZE(row)) {
            PyObject *v = PyList_GET_ITEM(row, idx);
            Py_INCREF(v);
            return v;
        }
    }
    return PySequence_GetItem(row, idx);
}

static int dict_append_row(PyObject *dict, PyObject *key, PyObject *row) {
    PyObject *grp = PyDict_GetItemWithError(dict, key);
    if (!grp) {
        if (PyErr_Occurred()) return -1;
        grp = PyList_New(1);
        if (!grp) return -1;
        Py_INCREF(row);
        PyList_SET_ITEM(grp, 0, row);
        int rc = PyDict_SetItem(dict, key, grp);
        Py_DECREF(grp);
        return rc;
    }
    return PyList_Append(grp, row);
}

/* Exact inner hash join on 1 integer key index (preserves first-seen source key-group order). */
static PyObject *inner_hash_join_1(PyObject *self, PyObject *args) {
    PyObject *s_rows, *j_rows;
    Py_ssize_t s_idx, j_idx;
    if (!PyArg_ParseTuple(args, "O!O!nn", &PyList_Type, &s_rows, &PyList_Type, &j_rows, &s_idx, &j_idx))
        return NULL;

    Py_ssize_t n_j = PyList_GET_SIZE(j_rows);
    Py_ssize_t n_s = PyList_GET_SIZE(s_rows);
    if (n_j == 0 || n_s == 0) return PyList_New(0);

    PyObject *join_map = PyDict_New();
    if (!join_map) return NULL;

    for (Py_ssize_t i = 0; i < n_j; i++) {
        PyObject *row = PyList_GET_ITEM(j_rows, i);
        PyObject *k = get_row_item(row, j_idx);
        if (!k) { Py_DECREF(join_map); return NULL; }
        if (k != Py_None) {
            if (dict_append_row(join_map, k, row) < 0) {
                Py_DECREF(k);
                Py_DECREF(join_map);
                return NULL;
            }
        }
        Py_DECREF(k);
    }

    if (PyDict_GET_SIZE(join_map) == 0) {
        Py_DECREF(join_map);
        return PyList_New(0);
    }

    PyObject *src_groups = PyDict_New();
    if (!src_groups) { Py_DECREF(join_map); return NULL; }

    for (Py_ssize_t i = 0; i < n_s; i++) {
        PyObject *row = PyList_GET_ITEM(s_rows, i);
        PyObject *k = get_row_item(row, s_idx);
        if (!k) { Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }
        if (k != Py_None) {
            PyObject *jg = PyDict_GetItemWithError(join_map, k);
            if (!jg) {
                if (PyErr_Occurred()) {
                    Py_DECREF(k); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
                }
            } else {
                if (dict_append_row(src_groups, k, row) < 0) {
                    Py_DECREF(k); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
                }
            }
        }
        Py_DECREF(k);
    }

    Py_ssize_t total_out = 0;
    PyObject *k, *sg;
    Py_ssize_t pos = 0;
    while (PyDict_Next(src_groups, &pos, &k, &sg)) {
        PyObject *jg = PyDict_GetItem(join_map, k);
        if (jg) total_out += PyList_GET_SIZE(sg) * PyList_GET_SIZE(jg);
    }

    PyObject *out = PyList_New(total_out);
    if (!out) { Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }

    Py_ssize_t out_i = 0;
    pos = 0;
    while (PyDict_Next(src_groups, &pos, &k, &sg)) {
        PyObject *jg = PyDict_GetItem(join_map, k);
        if (!jg) continue;
        Py_ssize_t n_sg = PyList_GET_SIZE(sg);
        Py_ssize_t n_jg = PyList_GET_SIZE(jg);
        for (Py_ssize_t a = 0; a < n_sg; a++) {
            PyObject *sr = PyList_GET_ITEM(sg, a);
            for (Py_ssize_t b = 0; b < n_jg; b++) {
                PyObject *jr = PyList_GET_ITEM(jg, b);
                PyObject *joined = concat_tuples(sr, jr);
                if (!joined) {
                    Py_DECREF(out); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
                }
                PyList_SET_ITEM(out, out_i++, joined);
            }
        }
    }

    Py_DECREF(src_groups);
    Py_DECREF(join_map);
    return out;
}

/* Exact inner hash join on 2 integer key indices. */
static PyObject *inner_hash_join_2(PyObject *self, PyObject *args) {
    PyObject *s_rows, *j_rows;
    Py_ssize_t s0, s1, j0, j1;
    if (!PyArg_ParseTuple(args, "O!O!nnnn", &PyList_Type, &s_rows, &PyList_Type, &j_rows, &s0, &s1, &j0, &j1))
        return NULL;

    Py_ssize_t n_j = PyList_GET_SIZE(j_rows);
    Py_ssize_t n_s = PyList_GET_SIZE(s_rows);
    if (n_j == 0 || n_s == 0) return PyList_New(0);

    PyObject *join_map = PyDict_New();
    if (!join_map) return NULL;

    for (Py_ssize_t i = 0; i < n_j; i++) {
        PyObject *row = PyList_GET_ITEM(j_rows, i);
        PyObject *v0 = get_row_item(row, j0);
        if (!v0) { Py_DECREF(join_map); return NULL; }
        if (v0 == Py_None) { Py_DECREF(v0); continue; }
        PyObject *v1 = get_row_item(row, j1);
        if (!v1) { Py_DECREF(v0); Py_DECREF(join_map); return NULL; }
        if (v1 == Py_None) { Py_DECREF(v0); Py_DECREF(v1); continue; }
        PyObject *k = PyTuple_New(2);
        if (!k) { Py_DECREF(v0); Py_DECREF(v1); Py_DECREF(join_map); return NULL; }
        PyTuple_SET_ITEM(k, 0, v0);
        PyTuple_SET_ITEM(k, 1, v1);
        if (dict_append_row(join_map, k, row) < 0) {
            Py_DECREF(k); Py_DECREF(join_map); return NULL;
        }
        Py_DECREF(k);
    }

    if (PyDict_GET_SIZE(join_map) == 0) {
        Py_DECREF(join_map);
        return PyList_New(0);
    }

    PyObject *src_groups = PyDict_New();
    if (!src_groups) { Py_DECREF(join_map); return NULL; }

    for (Py_ssize_t i = 0; i < n_s; i++) {
        PyObject *row = PyList_GET_ITEM(s_rows, i);
        PyObject *v0 = get_row_item(row, s0);
        if (!v0) { Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }
        if (v0 == Py_None) { Py_DECREF(v0); continue; }
        PyObject *v1 = get_row_item(row, s1);
        if (!v1) { Py_DECREF(v0); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }
        if (v1 == Py_None) { Py_DECREF(v0); Py_DECREF(v1); continue; }
        PyObject *k = PyTuple_New(2);
        if (!k) { Py_DECREF(v0); Py_DECREF(v1); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }
        PyTuple_SET_ITEM(k, 0, v0);
        PyTuple_SET_ITEM(k, 1, v1);
        PyObject *jg = PyDict_GetItemWithError(join_map, k);
        if (!jg) {
            if (PyErr_Occurred()) {
                Py_DECREF(k); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
            }
        } else {
            if (dict_append_row(src_groups, k, row) < 0) {
                Py_DECREF(k); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
            }
        }
        Py_DECREF(k);
    }

    Py_ssize_t total_out = 0;
    PyObject *k, *sg;
    Py_ssize_t pos = 0;
    while (PyDict_Next(src_groups, &pos, &k, &sg)) {
        PyObject *jg = PyDict_GetItem(join_map, k);
        if (jg) total_out += PyList_GET_SIZE(sg) * PyList_GET_SIZE(jg);
    }

    PyObject *out = PyList_New(total_out);
    if (!out) { Py_DECREF(src_groups); Py_DECREF(join_map); return NULL; }

    Py_ssize_t out_i = 0;
    pos = 0;
    while (PyDict_Next(src_groups, &pos, &k, &sg)) {
        PyObject *jg = PyDict_GetItem(join_map, k);
        if (!jg) continue;
        Py_ssize_t n_sg = PyList_GET_SIZE(sg);
        Py_ssize_t n_jg = PyList_GET_SIZE(jg);
        for (Py_ssize_t a = 0; a < n_sg; a++) {
            PyObject *sr = PyList_GET_ITEM(sg, a);
            for (Py_ssize_t b = 0; b < n_jg; b++) {
                PyObject *jr = PyList_GET_ITEM(jg, b);
                PyObject *joined = concat_tuples(sr, jr);
                if (!joined) {
                    Py_DECREF(out); Py_DECREF(src_groups); Py_DECREF(join_map); return NULL;
                }
                PyList_SET_ITEM(out, out_i++, joined);
            }
        }
    }

    Py_DECREF(src_groups);
    Py_DECREF(join_map);
    return out;
}

/* Pure C column-index tuple projection: [tuple(r[i] for i in idxs) for r in rows] */
static PyObject *project_idxs(PyObject *self, PyObject *args) {
    PyObject *rows, *idxs_obj;
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &rows, &PyTuple_Type, &idxs_obj))
        return NULL;

    Py_ssize_t n_cols = PyTuple_GET_SIZE(idxs_obj);
    Py_ssize_t n_rows = PyList_GET_SIZE(rows);
    Py_ssize_t stack_idxs[64];
    Py_ssize_t *idxs = (n_cols <= 64) ? stack_idxs : (Py_ssize_t *)PyMem_Malloc(sizeof(Py_ssize_t) * n_cols);
    if (!idxs) return PyErr_NoMemory();

    for (Py_ssize_t c = 0; c < n_cols; c++) {
        idxs[c] = PyLong_AsSsize_t(PyTuple_GET_ITEM(idxs_obj, c));
        if (idxs[c] == -1 && PyErr_Occurred()) {
            if (idxs != stack_idxs) PyMem_Free(idxs);
            return NULL;
        }
    }

    PyObject *out = PyList_New(n_rows);
    if (!out) {
        if (idxs != stack_idxs) PyMem_Free(idxs);
        return NULL;
    }

    for (Py_ssize_t r = 0; r < n_rows; r++) {
        PyObject *row = PyList_GET_ITEM(rows, r);
        PyObject *t = PyTuple_New(n_cols);
        if (!t) {
            Py_DECREF(out);
            if (idxs != stack_idxs) PyMem_Free(idxs);
            return NULL;
        }
        for (Py_ssize_t c = 0; c < n_cols; c++) {
            PyObject *v = get_row_item(row, idxs[c]);
            if (!v) {
                Py_DECREF(t); Py_DECREF(out);
                if (idxs != stack_idxs) PyMem_Free(idxs);
                return NULL;
            }
            PyTuple_SET_ITEM(t, c, v);
        }
        PyList_SET_ITEM(out, r, t);
    }

    if (idxs != stack_idxs) PyMem_Free(idxs);
    return out;
}

static PyMethodDef FastJoinMethods[] = {
    {"inner_hash_join_1", inner_hash_join_1, METH_VARARGS, "Fast 1-key inner hash join"},
    {"inner_hash_join_2", inner_hash_join_2, METH_VARARGS, "Fast 2-key inner hash join"},
    {"project_idxs", project_idxs, METH_VARARGS, "Fast column-index tuple projection"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastjoinmodule = {
    PyModuleDef_HEAD_INIT,
    "_fast_join_c",
    NULL,
    -1,
    FastJoinMethods
};

PyMODINIT_FUNC PyInit__fast_join_c(void) {
    return PyModule_Create(&fastjoinmodule);
}
