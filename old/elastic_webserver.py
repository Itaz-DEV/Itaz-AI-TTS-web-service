from waitress import serve
from elastic_app import app as application
import os
# set current working directory to source code directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("\nCurrent working directory: ---'{}'---\n".format(os.getcwd()))

if __name__ == '__main__':
   # serve(application, host='0.0.0.0')
   application.run(host='0.0.0.0')
