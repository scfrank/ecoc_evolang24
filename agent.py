"""Iterated Learning agent with a codebook."""

from collections import defaultdict

import numpy as np

from codebook import Codebook


class CBAgent(Codebook):

    def get_row_reps(self):
        """Return current cookbook rows aka data representations."""
        return self.data_reps

    def update_rows(self, row_labels):
        """Rows with identical codewords get same cluster assignment/label.

        Merge rows based on the cluster assignments in row_labels (e.g. identical rows have the same label.)
        New representation of the class is based on current classifiers.

        row_labels labels should be 0...number_of_clusters/rows.
        NB that this can be iterative, and rows can already have been merged: len(row_labels) != len(self.data)

        Need to change:
            data_to_rows map, rows_data_idx
            update codebook - run classifiers

        """
        new_rows = defaultdict(list)  #  new rows_data_idx
        for r, l in enumerate(row_labels):  # r: current row, l: new row
            row_items = self.rows_data_idx[r]  # current assignment of items to row r
            new_rows[l].extend(row_items)
            for i in row_items:
                self.data_to_rows_map[i] = l  # update to new row/label
        self.rows_data_idx = new_rows

        reps = self.get_row_reps()

        # update cookbook by prediction using classifiers
        # XXX dubious: Use all classifiers here, b/c unused classifiers will be masked; but this keeps indexing constant.
        code_columns = []
        # print(len(self.classifiers), 'of which disc in use', len(self.classifiers_in_use))
        #for clf_idx in self.classifiers_in_use:
        for clf_idx in range(len(self.classifiers)):
            if clf_idx in self.classifiers_in_use:
                clf = self.classifiers[clf_idx]
                col_prediction = clf.predict(reps)
            else:
                col_prediction = np.zeros((len(reps)))
            code_columns.append(col_prediction)

        self.codebook = np.array(code_columns).transpose()
