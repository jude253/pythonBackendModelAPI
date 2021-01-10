# I am not sure that this file is necessary or that it helps

import os
from route_config import *
app.debug = True
host = os.environ.get('IP', '0.0.0.0')
port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
    app.run(host=host)
