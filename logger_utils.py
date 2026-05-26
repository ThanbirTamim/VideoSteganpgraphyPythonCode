import os
from datetime import datetime


def ensure_logs():

    os.makedirs("logs", exist_ok=True)


def write_log(filename, text):

    ensure_logs()

    path = os.path.join("logs", filename)

    with open(path, "a", encoding="utf-8") as f:

        f.write(text + "\n")


def log_header(filename, title):

    line = "=" * 80

    text = f"""

{line}
{title}
TIME: {datetime.now()}
{line}

"""

    write_log(filename, text)