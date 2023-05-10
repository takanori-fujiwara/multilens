## Python implementation of Multi-LENS

About
-----
* Python3 implemetation of Multi-LENS.
  * Jin et al., Latent Network Summarization: Bridging Network Embedding and Summarization, KDD, 2019.
  * This implementation using graph-tool provides more flexibility in learning settings (e.g., base features, etc).
******

Requirements
-----
* Python3
* graph-tool (https://graph-tool.skewed.de/)
* Note: Tested on macOS Ventura.
******

Setup
-----
* Install graph-tool (https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions)
  * For example, macOS with Homebrew,

    `brew install graph-tool`

* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

******

Usage
-----
* Import installed modules from python (e.g., `from multilens import MultiLens`). See sample.py for examples.
