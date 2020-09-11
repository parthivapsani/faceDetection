pythonMax = max  # Store inbuilt max function before OpenCV import overrides inbuilt max function
defaultTFT = 'xlsx'

importModules = ['sys', 'argparse', 'os', 'shutil', 'time', 'xlsxwriter', 'cv2', 'signal']
for module in importModules:
    try:
        globals()[module] = __import__(module)
    except ImportError:
        if module == 'xlsxwriter':
            print("\nxlsxwriter not installed, transcript file type defaulting to csv")
            defaultTFT = 'csv'
        else:
            print("'{}' was not successfully imported, please install '{}' and try again".format(module, module))
            quit()
    except Exception as exception:
        print("{} exception was thrown when trying to import '{}'".format(exception, module))
        quit()

from utils import *

#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yoloface.cfg',
                    help='path to config file')
parser.add_argument('--model-weights', type=str, default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')
parser.add_argument('--video', '-v', type=str, default='',
                    help='path to video file')
parser.add_argument('--output-dir', '-o', type=str, default='FaceDetection/',
                    help='path to the output directory')
parser.add_argument('--output-file-name', '-f', type=str, default='YoloFace_Output',
                    help='file name for the output video')
parser.add_argument('--transcript-file-type', '-tft', type=str, default=defaultTFT,
                    help='file type of the produced frame transcript')
parser.add_argument('--rotation-angle', '-r', type=int, default=0,
                    help='rotation angle for video file')
parser.add_argument('--live', '-l', action='store_true',
                    help='changing source of the video to a webcam')
parser.add_argument('--get-help', '-gh', action='store_true',
                    help='argument used to bring up argument descriptors')
parser.add_argument('--demo', '-d', action='store_true',
                    help='flag for demo run (this deletes unnecessary folders and files)')
args = parser.parse_args()

#####################################################################
# Performs initial argument checking
if args.get_help:
    print("\nValid arguments were not provided, please refer to the following description for information on proper usage\n")
    print("'--model-cfg' is used to configure the convolutional neural network configuration used for facial detection")
    print("'--model-weights' is used to modify the convolutional neural network weights used for facial detection\n")
    print("'--output-dir' or '-o' is used to determine which output directory to use")
    print("Please note that providing an existing directory name will clear the contents of that directory")
    print("If no argument is provided, a directory named 'outputs' will be created for output storage\n")
    print("'--output-file-name' or '-f' is used to name the output file for the video")
    print("If no argument is provided, the output video will be named 'YoloFace_Output'\n")
    print("'--transcript-file-type' or '-tft' is used to determine the file type for the optional frame transcript")
    print("Please note that if no argument is provided, the frame transcript will be in xlsx format (or csv if xlsxwriter is not imported)\n")
    print("'--rotation-angle' or '-r' is used to determine how much to rotate the video")
    print("Please note that the angle is clockwise and that there will be a textual aid and preview available to assist this process")
    print("If a multiple of 90° is not provided, the setup process will appear to properly rotate the frame")
    print("The system will attempt to properly orient the video beforehand, but requires user confirmation before continuing\n")
    print('#' * 60)
    print("\n'--video' or '-v' is used to get the relative file path for the input video")
    print("'--live' or '-l' is used to set the video input to the current device's webcam")
    print("Only one of these two arguments is required for program operation, if both are provided, live recording will take precedence\n")
    print("'--demo' or '-d' is used for providing a demo (this changes folder and file construction behavior)")
    sys.exit(0)
elif not (args.live and args.video):  # Default to live demo if no video source argument is provided
    args.live = True

if args.live:  # Camera checks
    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        sys.exit("\nUnable to open on-device camera\n")
    windowName = "Live Demo of Face Detection using YoloFace"
    print('\nVideo source: on-device camera')
else:  # Video file checks
    if not os.path.isfile(args.video):
        sys.exit("\nInput video file {} doesn't exist".format(args.video))
    cap = cv2.VideoCapture(args.video)
    if cap is None or not cap.isOpened():
        sys.exit("\nUnable to open input video file: {}".format(args.video))
    windowName = "Face Detection using YoloFace"
    print('\nPath to video file:', args.video)
args.transcript_file_type = args.transcript_file_type.strip().lower()
if args.transcript_file_type not in ['txt', 'csv', 'xlsx']:
    print("'{}' was not an acceptable filetype, filetype is reverting to {}".format(args.transcript_file_type, defaultTFT))
    args.transcript_file_type = defaultTFT
print('Transcript file type:', args.transcript_file_type)

# Checks if weights file is present
if not os.path.isfile(args.model_weights) or not str(args.model_weights).endswith(".weights"):
    print("The provided weights file '{}' is not present, defaulting to {}".format(args.model_weights, './model-weights/yolov3-wider_16000.weights'))
    args.model_weights = './model-weights/yolov3-wider_16000.weights'
    if not os.path.isfile(args.model_weights) or not str(args.model_weights).endswith(".weights"):
        sys.exit("\nWeights are not present, please run the 'get_models.sh' file or provide a valid weights file\n")
else:
    print('The weights of model file:', args.model_weights)

# Checks if the config file is present
if not os.path.isfile(args.model_cfg) or not str(args.model_cfg).endswith(".cfg"):
    print("Provided config file '{}' is not valid, defaulting to {}".format(args.model_cfg, './cfg/yoloface.cfg'))
else:
    print('The weights of model file:', args.model_weights)

# Check outputs directory
if not os.path.exists(args.output_dir):
    print('\nCreating the {} directory'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    shutil.rmtree(args.output_dir)  # Deletes the existing directory
    os.makedirs(args.output_dir)  # Recreates the directory
    print('\nDeleted the existing {} directory and created a new one'.format(args.output_dir))
print('Path to output directory:', args.output_dir)

# Give the configuration and weight files for the model and load the network using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

lastPosition, lastConfidence = (-1, -1), -1  # Negative values used to bypass initial iteration logic
width, height, fps = -1, -1, -1  # Negative values used to bypass initial iteration logic
confidenceDropOff = 0.05  # % Deviation of confidence estimation between sequential frames
frameCounter = 0  # Frame counter
totalDimensions = list()  # Global frame dimension storage
interrupt = False  # Global keyboard interrupt listener flag

processDirPath = './VideoWriter/cmake-build-debug/temporary_image_storage'
preProcessDirPath = './VideoWriter/cmake-build-debug/temporary_image_storage_with_overlay'
preProcessDirPath2 = './VideoWriter/cmake-build-debug/temporary_image_storage_full_frame'
for preDirPath in [processDirPath, preProcessDirPath, preProcessDirPath2]:
    if not os.path.exists(preDirPath):
        os.makedirs(preDirPath)
    else:
        shutil.rmtree(preDirPath)  # Deletes the existing directory
        os.makedirs(preDirPath)  # Recreates the directory
print("\n", '#' * 60, "\n")


def interruptHandler(signal, frame):
    print("\nKeyboard interrupt detected, stopping processing now\n")
    global interrupt
    interrupt = True


def rotateFrame(mat, angle):  # Video inputs are rotated 90° counterclockwise for some reason, this rotates each frame
    matHeight, matWidth = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (matWidth / 2, matHeight / 2)  # getRotationMatrix2D needs coordinates in reverse order (matWidth, matHeight) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new matWidth and matHeight bounds
    bound_w = int(matHeight * abs_sin + matWidth * abs_cos)
    bound_h = int(matHeight * abs_cos + matWidth * abs_sin)

    # subtract old image center (bringing image back to the origin) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def processImage(frame, count, angle, dirPath, dirPath2):
    output_file = "frame_{}.png".format(count)
    if angle:
        frame = rotateFrame(frame, angle)
    originalFrame = frame.copy()
    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    # Remove the bounding boxes with low confidence
    faces, dimensions = processFaces(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    cv2.putText(frame, "Frame: {}".format(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, black, 2)
    cv2.putText(frame, "Faces: {}".format(len(faces)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, black, 2)
    sys.stdout.write("\rFrame {} ==> # of Detected Faces: {}".format(count, len(faces)))
    sys.stdout.flush()
    # Saves frame if param specifies as such
    if dirPath and os.path.exists(dirPath):
        cv2.imwrite(os.path.join(dirPath, output_file), frame.astype(np.uint8))
    if dirPath2 and os.path.exists(dirPath2):
        cv2.imwrite(os.path.join(dirPath2, output_file), originalFrame.astype(np.uint8))
    return frame, dimensions


def getRotationAngle(originalFrame, angle=0):
    if angle:
        frame = rotateFrame(originalFrame, angle)
    elif angle == 360:
        sys.exit("No faces were detected in the first frame")
    else:
        frame = originalFrame
    # Creates the DNN
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    # Runs the forward pass to get output of the output layers
    outs = net.forward(get_outputs_names(net))
    # Remove the bounding boxes with low confidence
    faces, dimensions = processFaces(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    # Change the angle if faces were not detected
    if not len(faces):
        return getRotationAngle(originalFrame, angle + 90)
    # Shows the user a preview of the system generated rotation angle
    cv2.namedWindow("Preview Frame", cv2.WINDOW_NORMAL)
    cv2.imshow("Preview Frame", frame.astype(np.uint8))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    while True:  # Gets user input
        print("\nThe image will be rotated counterclockwise {} degrees, please check the preview for confirmation".format(angle))
        print("If this {} degree rotation angle is okay, press enter at the next step. Otherwise, please enter a new rotation angle".format(angle))
        val = input("If the rotation angle is unsatisfactory, provide one here. Otherwise, press 'Enter': ")
        # Checks if the user entered a value
        if val:
            try:
                # Performs type checking
                angle = abs(int(val))
                if angle % 90:
                    print("Invalid value entered, please provide another angle (it must be a positive integer multiple of 90)")
                    continue
                angle %= 360
                print("Rotation Angle: {} degrees\n".format(angle))
                cv2.destroyAllWindows()
                break
            except ValueError:
                print("Invalid value entered, please provide another angle (it must be an integer value)")
        else:
            print("Rotation Angle: {} degrees\n".format(angle))
            break
    cv2.destroyAllWindows()
    return angle


def getBestFace(faces):  # Picks the best face in each frame
    global lastPosition, lastConfidence
    if len(faces) == 1:
        return faces[0][1:], faces[0][3]
    positions = [(face[3], face[4]) for face in faces]  # X, Y topLeft
    confidences = [(face[0], index) for index, face in enumerate(faces)]  # Confidence, index
    if lastPosition != (-1, -1):
        distances = [(((lastPosition[0] - position[0]) ** 2 + (lastPosition[1] - position[1]) ** 2) ** 0.5, index) for index, position in enumerate(positions)]
        distances.sort()
        for distance in distances:
            if confidences[distance[1]][0] + confidenceDropOff > lastConfidence:
                lastPosition, lastConfidence = positions[distance[1]], confidences[distance[1]][0]
                return faces[distance[1]][1:], lastConfidence
    else:
        centerX, centerY = width / 2, height / 2
        distances = [(((centerX - position[0]) ** 2 + (centerY - position[1]) ** 2) ** 0.5, index) for index, position in enumerate(positions)]
    # Either first frame or confidence drop off was too steep
    distanceCopy, confidenceCopy = distances[:], confidences[:]
    distances.sort()  # Sorts distances in ascending order
    confidences.sort(reverse=True)  # Sorts confidences in descending order
    rankings = list()
    for index in range(len(faces)):
        rankings.append((distances.index(distanceCopy[index]) + confidences.index(confidenceCopy[index]), index))
    rankings.sort()  # Sorts rankings by least total score (low scores are desirable like golf)
    if rankings[0][0] == rankings[1][0]:
        # Indicates a tie, will resort solely by distance to center
        tiedIndices, tiedScore = list(), rankings[0][0]
        for index, ranking in enumerate(rankings):
            if ranking[0] == tiedScore:
                tiedIndices.append(index)
        rankedTies = [distanceCopy[index] for index in tiedIndices]
        rankedTies.sort()
        bestIndex = rankedTies[0][1]
    else:
        bestIndex = rankings[0][1]
    lastPosition, lastConfidence = positions[bestIndex], confidences[bestIndex][0]
    return faces[bestIndex][1:], lastConfidence


def processFrameData():
    xlsx = False
    row, outputFileName, contentFormat, workbook, transcript, outputFile = '', '', '', '', '', ''
    if args.transcript_file_type == 'xlsx':
        xlsx = True
        outputFileName = 'frame_transcript.xlsx'
        workbook = xlsxwriter.Workbook(os.path.join(args.output_dir, outputFileName))
        transcript = workbook.add_worksheet('Frame Data')
        row = 0
        transcript.set_column('D:D', 13)
        titleFormat = workbook.add_format({'bold': True, 'align': 'center'})
        contentFormat = workbook.add_format({'align': 'center'})
        transcript.write(row, 0, "Frame #", titleFormat)
        transcript.write(row, 1, "# of Faces", titleFormat)
        transcript.write(row, 2, "Face #", titleFormat)
        transcript.write(row, 3, "Confidence", titleFormat)
        transcript.write(row, 4, "Width", titleFormat)
        transcript.write(row, 5, "Height", titleFormat)
        transcript.write(row, 6, "Left", titleFormat)
        transcript.write(row, 7, "Top", titleFormat)
        transcript.write(row, 8, "Right", titleFormat)
        transcript.write(row, 9, "Bottom", titleFormat)
    else:
        outputFileName = "frame_transcript." + args.transcript_file_type
        outputFile = open(os.path.join(args.output_dir, outputFileName), "w+")
        outputFile.write("Frame #,# of Faces,Face #,Confidence,Width,Height,Left,Top,Right,Bottom\n")  # 10 columns
    faceData, faceWidth, faceHeight = list(), list(), list()
    for frameNum, frame in enumerate(totalDimensions):
        if len(frame):  # If there is one or more faces in the current frame, iterate through
            bestFrameFace, bestConfidence = getBestFace(frame)
            if xlsx:
                row += 1
                transcript.write(row, 0, '')
            else:
                outputFile.write('\n')
            for faceNum, face in enumerate(frame):  # Iterates through the frame data for each face
                if xlsx:
                    row += 1
                    face = [frameNum + 1, len(frame), faceNum + 1] + face
                    for col, data in enumerate(face):
                        transcript.write(row, col, data, contentFormat)
                else:
                    outputFile.write("{},{},{},{},{},{},{},{},{},{}\n".format(frameNum + 1, len(frame), faceNum + 1, face[0], face[1], face[2], face[3], face[4], face[5], face[6]))
            # Stores box data for best face
            faceData.append(bestFrameFace)
            faceWidth.append(bestFrameFace[0])
            faceHeight.append(bestFrameFace[1])
        # Saves record of empty frame to file
        else:
            if xlsx:
                row += 1
                transcript.write(row, 0, '')
                row += 1
                transcript.write(row, 0, frameNum + 1, contentFormat)
                transcript.write(row, 1, 0, contentFormat)
            else:
                outputFile.write("{},{}".format(frameNum + 1, 0))
    print("Transcript was saved to", outputFileName)
    if xlsx:
        workbook.close()
    else:
        outputFile.close()
    # Returns maximum frame size for the second cropping stage
    return pythonMax(faceWidth), pythonMax(faceHeight), faceData


def getMaxFrameSize(maxWidth, maxHeight, faceData):
    frameBounds, frameChanges = [width, height], [round(maxWidth / 2), round(maxHeight / 2)]
    originalWidth, originalHeight = maxWidth, maxHeight  # Stores originals to see if max bounds were changed during iteration
    newFaceData = list()
    for face in faceData:  # Iterating through each primary face
        centerX, centerY = ((face[2] + face[4]) / 2), ((face[3] + face[5]) / 2)  # X, Y of center
        topLeft, bottomRight = [centerX - frameChanges[0], centerY - frameChanges[1]], [centerX + frameChanges[0], centerY + frameChanges[1]]
        # X, Y of topLeft and bottomRight corners (because if the face is not the
        boundsCheck = [-1 - topLeft[0], -1 - topLeft[1], bottomRight[0] - width, bottomRight[1] - height]
        # Checks if any part of the newly expanded face is outside the frame by comparing to the outer bounds of the frame
        for index in range(4):
            if boundsCheck[index] > -1:
                if index % 2:  # new maxHeight (odd indices)
                    maxHeight = round(maxHeight - 2 * boundsCheck[index])
                else:  # new maxWidth (even indices)
                    maxWidth = round(maxWidth - 2 * boundsCheck[index])
                frameChanges = [round(maxWidth / 2), round(maxHeight / 2)]  # Updates the frame change values
        newFaceData.append(topLeft)  # Only need to store topLeft because maxWidth and maxHeight will be used for frame cropping
    if maxWidth != originalWidth or maxHeight != originalHeight:
        return getMaxFrameSize(maxWidth, maxHeight, faceData)
    return maxWidth, maxHeight, faceData


def cropImage(frame, count, angle, dirPath, frameWidth, frameHeight, topLeft):
    output_file = "frame_{}.png".format(count)
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    if angle:
        frame = rotateFrame(frame, angle)
    newFrame = frame[topLeft[1]:topLeft[1] + frameHeight, topLeft[0]:topLeft[0] + frameWidth].copy()
    cv2.imwrite(os.path.join(dirPath, output_file), newFrame.astype(np.uint8))


def main():
    signal.signal(signal.SIGINT, interruptHandler)
    global width, height, fps, frameCounter, totalDimensions
    rotationAngle = False
    width, height, fps = round(cap.get(3)), round(cap.get(4)), round(cap.get(5))
    print("\nDimensions of input video: {} x {}".format(width, height))
    if not args.live:
        print("FPS of Input Video: {}\n".format(fps))
        if args.rotation_angle and args.rotation_angle % 90 == 0:
            rotationAngle = args.rotation_angle
    hasFaces = list()
    startTime = time.time()
    while True:
        has_frame, frame = cap.read()
        if not has_frame:  # Stop the program if reached end of video
            print('\nInitial Processing Complete!\n')
            cv2.waitKey(100)
            break
        if not args.live and rotationAngle is False and frameCounter == 0:  # Gets rotation angle for the frame if initial orientation is incorrect
            rotationAngle = getRotationAngle(frame.copy())
            cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        frameCounter += 1
        frame, dimensions = processImage(frame, frameCounter, rotationAngle, preProcessDirPath, preProcessDirPath2)  # Gets the new frame and position data for each face
        hasFaces.append(1) if dimensions else hasFaces.append(0)  # Checks if the current frame has a face
        totalDimensions.append(dimensions)  # Adds this data to the local position data list
        cv2.imshow(windowName, frame)  # Displays image
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q') or interrupt:
            print('\nFrame {} ==> Interrupted by user'.format(frameCounter))
            if args.live:
                print('Live Face Detection Complete\n')
            break
    cap.release()  # Releases the video writer
    cv2.destroyAllWindows()  # Closes the preview window
    usableFrames = sum(hasFaces)
    if args.live:
        fps = round(frameCounter / (time.time() - startTime), 3)
        print("Approximated FPS: {:.3f}\nDimensions: {} x {}\nNumber of Frames Processed: {}\nUsable Frames Processed: {}\n".format(fps, width, height, frameCounter, usableFrames))
        performPostProcessing = input("Please type in the phrase 'continue' to perform post processing, otherwise press any key to exit: ")
        if not (performPostProcessing and "continue" in performPostProcessing.strip().lower()):
            print("\nThanks for viewing this project demo!\n")
            return
    else:
        print("Approximated Computation FPS: {:.3f}\nNumber of Frames Processed: {}\nUsable Frames Processed: {}\n".format(round(frameCounter / (time.time() - startTime), 3), frameCounter, usableFrames))
    print("\n", '#' * 60, "\n")
    print("Processing initial data")
    frameWidth, frameHeight, faceData = processFrameData()
    print("\nInitial Frame Width: {}\tInitial Frame Height: {}".format(frameWidth, frameHeight))
    frameWidth, frameHeight, newFaceData = getMaxFrameSize(frameWidth, frameHeight, faceData)
    print("Final Frame Width: {}\t\tFinal Frame Height: {}\n".format(frameWidth, frameHeight))
    if args.live:
        indexCounter = 0
        for index in range(frameCounter):
            if hasFaces[index]:
                indexCounter += 1
                imgFileName = './VideoWriter/cmake-build-debug/temporary_image_storage_full_frame/frame_{}.png'.format(index + 1)
                cropImage(imgFileName, indexCounter, 0, processDirPath, frameWidth, frameHeight, newFaceData[indexCounter - 1][2:4])
                sys.stdout.write("\rFrame #{} post-processed".format(indexCounter))
                sys.stdout.flush()
        sys.stdout.write("\rAll frames post-processed")
        sys.stdout.flush()
        print("\n")
    else:
        cap2 = cv2.VideoCapture(args.video)
        for index in range(frameCounter):
            has_frame, frame = cap2.read()
            cropImage(frame, index + 1, rotationAngle, processDirPath, frameWidth, frameHeight, newFaceData[index][2:4])
        cap2.release()
    print("Finished Video Reprocessing and primary face cropping")
    print("Paste the following command into the command line to stitch the images together into a video")
    arguments = ['./VideoWriter/cmake-build-debug/VideoWriter', str(frameCounter), str(frameWidth), str(frameHeight), str(fps), os.path.join(args.output_dir, args.output_file_name)]
    print("\n{}\n".format(' '.join(arguments)))
    if args.live:
        print("Thanks for viewing this project demo!\n")
    if args.demo:  # Delete unnecessary folders and files
        deleteDirectories = [args.output_dir, processDirPath, preProcessDirPath, preProcessDirPath2]
        for directory in deleteDirectories:
            print(directory)
            shutil.rmtree(directory)


if __name__ == '__main__':
    main()

# Sample Input
# python3 yoloface.py -v samples/TestVid4.MOV -o multiFaceTest -r 0

# Live Demo Input (make sure to stash and checkout to master first)
# python3 yoloface.py -v samples/TestVid4.MOV -o liveDemo -r 0 -l True

# Shortened Sample Input
# python3 yoloface.py -v samples/TestVid4.MOV -o multiFaceTest -r 0 -s True
