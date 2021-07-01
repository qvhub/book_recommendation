# Rcommendation book system

## Installation

### Data Base Package

```bash

source .venv/bin/activate

pip install -r /path/to/requirements.txt

```

### Data Base

In folder api

```bash

mysql -u 'your_user_name' -p

'your_passeword'

use 'data_base_name';

source goodreads.sql;

```

## Usage

### Run server

In folder api run

```bash
python3 main.py
```
waiting...

```bash
INFO:    Uvicorn running on http://127.0.0.1:8001
```