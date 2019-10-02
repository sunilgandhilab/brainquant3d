# Install:
```
pip3 install --user 
```

After installation you should edit the packages settings for your system. The settings are help in:
```
clearmap3_directory>default.conf
```

# To generate html from docs:

from the 'docs/' directory:
```
sphinx-apidoc  -f -o source/ ../clearmap3/
make html
```
