import os
import sys
import fileinput

def doFile(filePath, textToSearch, textToReplace) :

    tempFile = open( filePath, 'r+' )

    for line in fileinput.input( filePath ):
        if textToSearch in line :
            tempFile.write( line.replace( textToSearch, textToReplace ) )
        else :
            tempFile.write( line )
    tempFile.close()


def main():
    image_path = '/opt/test/210969'
    image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    for image_file in image_files  :
        doFile()



main()
