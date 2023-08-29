from . import readData, constants

def fetch(datatype, numTraining):
    if datatype=='d':
        numTest = constants.DIGITS_TEST_DATA_SIZE
        legalLabels = range(10)

        rawTrainingData, chosenList = readData.loadDataFile(
            "data/digitdata/trainingimages", 
            numTraining, 
            constants.DIGIT_DATUM_WIDTH,
            constants.DIGIT_DATUM_HEIGHT,
            True
        )
    
        trainingLabels = readData.loadLabelsFile(
            "data/digitdata/traininglabels", 
            chosenList
        )

        # rawValidationData, chosenList = readData.loadDataFile(
        #     "data/digitdata/validationimages", 
        #     numTest, 
        #     constants.DIGIT_DATUM_WIDTH,
        #     constants.DIGIT_DATUM_HEIGHT
        # )

        # validationLabels = readData.loadLabelsFile(
        #     "data/digitdata/validationlabels", 
        #     chosenList
        # )
        
        rawTestData, chosenList = readData.loadDataFile(
            "data/digitdata/testimages", 
            numTest, 
            constants.DIGIT_DATUM_WIDTH, 
            constants.DIGIT_DATUM_HEIGHT
        
        )

        testLabels = readData.loadLabelsFile(
            "data/digitdata/testlabels", 
            chosenList
        )

    elif datatype=='f':
        numTest = constants.FACE_TEST_DATA_SIZE
        legalLabels = range(2)

        rawTrainingData, chosenList = readData.loadDataFile(
            "data/facedata/facedatatrain", 
            numTraining,
            constants.FACE_DATUM_WIDTH,
            constants.FACE_DATUM_HEIGHT, 
            True
        )

        trainingLabels = readData.loadLabelsFile(
            "data/facedata/facedatatrainlabels",  
            chosenList
        )

        # rawValidationData, chosenList = readData.loadDataFile(
        #     "data/facedata/facedatatrain", 
        #     numTest,
        #     constants.FACE_DATUM_WIDTH,
        #     constants.FACE_DATUM_HEIGHT
        # )

        # validationLabels = readData.loadLabelsFile(
        #     "data/facedata/facedatatrainlabels", 
        #     chosenList
        # )

        rawTestData, chosenList = readData.loadDataFile(
            "data/facedata/facedatatest", 
            numTest,
            constants.FACE_DATUM_WIDTH,
            constants.FACE_DATUM_HEIGHT
        )

        testLabels = readData.loadLabelsFile(
            "data/facedata/facedatatestlabels", 
            chosenList
        )
    else:
        return False

    return rawTrainingData, trainingLabels, rawTestData, testLabels, legalLabels