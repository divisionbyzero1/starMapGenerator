# starMapGenerator
The class file and example script utilize the GURPS First In rules to generate a boat-load of systems for a Traveller game.

# Why do it this way?
This set of routines was created so that one could literally draw an image of stellar densities and have the program populate it with procedural worlds.  It works!  The light and dark regions of the input maps correspond to sparse/rift to dense stellar areas.  After that, all of the system generation generally follows the GURPS First In rules.

The bit that isn't included in the First In rules are the ``empire stages'' that are included as a class method.  These are there to simulate the effect of jump technology allowing people to expand from their homeworld.  Along the way, they probably terraform the worlds as well to make them nicer places to live.  These are demonstrated in the example script.

# What do you do to get started?
Download or clone the repository.  There are two places where you will have to enter the PATH location of the files so when you run the script, it can find the class file and the class file can find some external files it needs.  Look for the ``sys.append.path()'' and the ``LOCALPATH'' statements right near the top of each file.  The example script is called ``Reddit_map.py'' and there is an example PNG image used that is 3x3 sectors large.  Run it and you are on the way.

# What are the caveats?
The input map has to be sized correctly for the Traveller sector system, which is in 32x40 parsec chunks.  In the input file, each pixel is a parsec, so that's how it goes.  The input also needs to be 8-bit grayscale or it won't know what to do with the other colors.

Not every single aspect of "First In" is repliated. Some of the details of eccentricity and tidal locking are skipped.  Some other short-cuts are made that could be fixed later on.

Most of the functions used are limited to within a single sector right now, so watch out when you want an empire to span two sectors or more.
