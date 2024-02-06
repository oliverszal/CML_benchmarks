This Repository currently features some files which let's you run basic benchmarks on D-Wave's annealer. 
The idea is to make your first steps on a Quantum computer.

For the files to work you'll need to open a new codespace (which requires a GitHub account). 
This generates a container on the GitHub cloud, which basically acts as its own operating system. 
Ideally, this should guarantee that you won't run into any software version specific problems.

To execute any of the .py or .ipynb files, you'll first need to paste in your D-Wave token into the file 'dwave_token.py'. 
To acquire such a token, you have to sign up to D-Waves cloud system "Leap" under https://cloud.dwavesys.com/leap/.
Make sure to not push the token to any live branch.

Heads up: Executing the .ipynb files requires choosing a Kernel. I recommend the standard recommended options, that is Python Environments... -> Python 3.11.7.

Step-by-step guide:
### Requirements:
- have a GitHub account
- run a codespace (green button with "<> Code")
- have a D-Wave Leap account with valid token (https://cloud.dwavesys.com/leap/)
- paste D-Wave token into the file 'dwave_token.py'
- When executing a ipynb file, choose Kernel Python Environments... -> Python 3.11.7
