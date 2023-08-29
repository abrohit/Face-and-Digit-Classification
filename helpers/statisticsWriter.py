"""
Given model statistics data and number of iterations the write function will write it into the specified text
file for each data type(digits and faces).
"""
from . import constants

def write(data, testIters, digitOutPath = 'output/perceptron_digit_results.txt', faceOutPath = 'output/perceptron_face_results.txt'):
        with open(digitOutPath, 'w') as f:
            for key, value in data['d'].items():
                f.write(f"{key}% where we have {key/100 * constants.DIGITS_TRAINING_DATA_SIZE} Training and {constants.DIGITS_TEST_DATA_SIZE} Testing.\n")
                f.write(f"Note: Averages were calculated with {testIters} iterations and rounded to 2 decimal places:\n")
                f.write(f"  Mean Accuracy: {round(value['mean']*100, 2)}% where we got {int(value['mean']*constants.DIGITS_TEST_DATA_SIZE)} correct out of {constants.DIGITS_TEST_DATA_SIZE}.\n")
                f.write(f"  Standard Deviation of Mean Accuracy: {round(value['std'], 2)}\n")
                f.write(f"  Average time of training: {round(value['avgTime']*100, 2)} seconds.\n")
                f.write(f"\n")
            f.close()

        with open(faceOutPath, 'w') as f:
            for key, value in data['f'].items():
                f.write(f"{key}% where we have {key/100 * constants.FACE_TRAINING_DATA_SIZE} Training and {constants.FACE_TEST_DATA_SIZE} Testing.\n")
                f.write(f"Note: Averages were calculated with {testIters} iterations and rounded to 2 decimal places:\n")
                f.write(f"  Mean Accuracy: {round(value['mean']*100, 2)}% where we got {int(value['mean']*constants.FACE_TEST_DATA_SIZE)} correct out of {constants.FACE_TEST_DATA_SIZE}.\n")
                f.write(f"  Standard Deviation of Mean Accuracy: {round(value['std'], 2)}\n")
                f.write(f"  Average time of training: {round(value['avgTime']*100, 2)} seconds.\n")
                f.write(f"\n")
            f.close()