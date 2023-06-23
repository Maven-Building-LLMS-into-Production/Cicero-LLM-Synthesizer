# MIT License
#
# Copyright (c) 2023 Victor Calderon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module containing the set of default variables of the project.
"""

# Option for saving the output data to disk
save_to_disk = True

# URL to the CICERO dataset
cicero_dataset_url = "https://raw.githubusercontent.com/hamzafarooq/maven-mlsystem-design-cohort-1/main/data/df_embed.csv"  # noqa: E501

# Option for saving to disk
save_to_disk = True

# Name of the column that corresponds to the Document ID
document_id_colname = "_id"

# Name of the column that corresponds to the title of the document.
title_colname = "title"

# Name of the column that contains the content of the document.
content_colname = "content"

# Name of teh target column name that will contain the parsed / clean version
# of the document's content.
clean_content_colname = "clean_content"