# DeepWell
Team repo for Wishing Well team - Virtual summer internship 2020


## Running with script (Windows/PowerShell)
First, open powershell and make sure you are allowed to run scripts by entering:

    Set-ExecutionPolicy RemoteSigned

You only have to do that one time.
Then you can navigate to the deepwell repository on your computer and build the container by running:

    .\deepwellstart.ps1 -build
Then, to run the container, simply enter:

    .\deepwellstart.ps1 -run

A handy tip is just to write the first couple of letters "de" and press tab to complete.
You can also combine the shortened version of the flags to do more things:
| Flag |  Action|
|--|--|
| -r |  Run|
| -br | Build + Run |
| -rs |  Run + Show|
|-brs|Build + Run + Show |

Show will open the browser and show the plot

If you still are not allowed to run the script, you can try this:

    Set-ExecutionPolicy Unrestricted

  

## Running without script (Non windows)

  
Instructions if you for some reason cannot launch docker using the script. 
(For example if you are on linux/mac)


First, navigate to the deepwell repository on your computer.

Then build the container using:

  

`docker build -t wishing-well-app .`

  

Then run it with:

  

`docker run -it --mount type=bind,source="$(pwd)",target=/app -p 8080:8080 --name wwrunning wishing-well-app`

  

It should now display the plot at http://localhost:8080/

  

When you have changed your code, you need to run

  

`docker rm -f wwrunning`

  

Then just run again.
  

Note that if you edit for example the env, you have to build the container again for the changes to take effect. If you are not seeing your changes being applied, try rebuilding.
