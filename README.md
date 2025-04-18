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
* Install with pip3. Move to the directory of this repository. Then,

    `pip3 install .`

* Install graph-tool (https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions)
  * For example, macOS with Homebrew (when not using virtual environment),

    `brew install graph-tool`
  
  * When using virtual environment, there are two options:
  
    - Option 1. Follow the graph-tool instruction (need a lot of time for compiling graph-tool).

      - Check a section of "Installing in a virtualenv".

      - graph-tool's instruction doesn't support Python3.12. For Python3.12, before the configure step (i.e., ./configure --prefix=$HOME/.local), run commands below:

        `pip3 install setuptools pycairo`

    - Option 2. Use virtual environment with "include-system-site-packages = true"
      
      - For example, edit "pyenv.cfg" to be `include-system-site-packages = true`

      - then
        
        `brew install graph-tool`

******

Usage
-----
* Import installed modules from python (e.g., `from multilens import MultiLens`). See sample.py for examples.
