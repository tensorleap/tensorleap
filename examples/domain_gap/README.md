# Tensorleap Quick Start

Tensorleap is a **debugging**, **observability**, and **explainability** platform for Deep Neural Networks that allows data scientists to develop advanced **Deep Learning** models dramatically **faster** and with far **better** results. This README will guide you through the steps to quickly get started and set up with TensorLeap.

This quick start guide will walk you through the steps to get started with Tensorleap CLI using this example repository.

For learning how to start a new Tensorleap project using CLI please refer to [#TODO]. This guide provides detailed instructions on starting a fresh Tensorleap project from the CLI.

## **Prerequisites**

Before you begin, ensure that you have the following prerequisites installed:

- **[Python](https://www.python.org/)** (version 3.7 or higher)
- **[Poetry](https://python-poetry.org/)**

When this repository is cloned to your local machine.

Change into the repository directory:

```bash 
cd repository
```


Install the project dependencies using Poetry:

```bash
poetry install
```

## Tensorleap **CLI Installation**

with `curl`:

```bash
curl -s https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh | bash
```

with `wget`:

```bash
wget -q -O - https://raw.githubusercontent.com/tensorleap/cli-go/master/install.sh | bash
```

CLI repository: https://github.com/tensorleap/cli-go

## Tensorleap CLI Usage

### Tensorleap **Login**

To login to Tensorealp: 

```bash
tensorleap auth login [api key] [api url].
```

- API Key is your Tensorleap token (see how to generate a CLI token in the section below).
- API URL is your Tensorleap environment URL: CLIENT_NAME.tensorleap.ai

**Generate CLI Token**

1. Login to CLIENT_NAME.tensorleap.ai
2. Scroll down to the bottom of the **Resources Management** page, then click `GENERATE CLI TOKEN`
3. Once a CLI token is generated, just copy and paste it in your shell: 

```
tensorleap auth login [api key] [api url]
```

## Tensorleap **Dataset Deployment**


To deploy your local changes:


```bash
tensorleap datasets save
```



### **Tensorleap files**

Tensorleap files in the repository include `tensorleap.py` and `.tensorleap.yaml`. The files consist of the  required configurations to make the code integrate with the Tensorleap engine:

**.tensorleap.yaml**

.tensorleap.yaml file is configured to a dataset in your Tensorleap environment and is synced to the dataset saved in the environment. 

For any additional file being used we add its path under `include` parameter:

```yaml
include:
  - tensorleap.py

```

**tensorleap.py file** 

`tensorleap.py` configure all binding functions used to bind to Tensorleap engine. These are the functions used to evaluate and train the model, visualize the variables, and enrich the analysis with external metadata variables

## Testing

To test the system we can run: 

[###TODO]

### Using **poetry**

**`tool.poetry.scripts`** section in the **`pyproject.toml`** file is used to define scripts or command-line utilities associated with your project. It allows you to specify custom commands that can be executed using the Poetry CLI.

The **`tool.poetry.scripts`** section is used to map a script name to a Python function or an executable command. When you define a script in this section, it becomes accessible as a command that can be executed from the command line using **`poetry run`**. 

You can add to **`pyproject.toml`**

```python
[tool.poetry.scripts]
my-test = "my_package.module:function"
```

After defining the script, you can execute it using the Poetry CLI:

```bash
poetry run my-test
```

This will execute the associated Python function or command.

*For further explanation please refer to the [docs](https://docs.tensorleap.ai/)*