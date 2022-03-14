import cv2


def load_cascade(label, path='../data/haarcascades/'):
    face_cascade_filenames = [
        'haarcascade_eye.xml',
        'haarcascade_lefteye_2splits.xml',
        'haarcascade_eye_tree_eyeglasses.xml',
        'haarcascade_licence_plate_rus_16stages.xml',
        'haarcascade_frontalcatface.xml',
        'haarcascade_lowerbody.xml',
        'haarcascade_frontalcatface_extended.xml',
        'haarcascade_profileface.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_righteye_2splits.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_russian_plate_number.xml',
        'haarcascade_frontalface_alt_tree.xml',
        'haarcascade_smile.xml',
        'haarcascade_frontalface_default.xml',
        'haarcascade_upperbody.xml',
        'haarcascade_fullbody.xml',
    ]

    filename = [f for f in face_cascade_filenames if label in f]
    if len(filename) == 0:
        raise Exception('no file with ' + label)
    elif len(filename) > 1:
        raise Exception('too many files with ' + label)

    face_cascade = cv2.CascadeClassifier()

    full_path = path + filename[0]

    if not face_cascade.load(cv2.samples.findFile(full_path)):
        raise Exception('error loading ' + label)

    return face_cascade


front_face = load_cascade('frontalface_default')
upper_body = load_cascade('upperbody')
full_body = load_cascade('fullbody')

blue = (255, 0, 0)

vid = cv2.VideoCapture(0)
ret, frame = vid.read()

while ret:

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = front_face.detectMultiScale(frame_grey)
    # uppers = upper_body.detectMultiScale(frame_grey)
    # fulls = full_body.detectMultiScale(frame_grey)
    # print(len(faces), len(uppers), len(fulls))

    for (x, y, w, h) in faces:
        frame = cv2.line(frame, (x, y), (x, y + h), blue, 3)
        frame = cv2.line(frame, (x, y), (x + w, y), blue, 3)
        frame = cv2.line(frame, (x + w, y + h), (x, y + h), blue, 3)
        frame = cv2.line(frame, (x + w, y + h), (x + w, y), blue, 3)

    # for (x, y, w, h) in uppers:
    #     center = (x + w // 2, y + h // 2)
    #     frame = cv2.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

    # for (x, y, w, h) in fulls:
    #     frame = cv2.line(frame, (x, y), (x, y + h), (255, 0, 0), 3)
    #     frame = cv2.line(frame, (x, y), (x + w, y), (255, 0, 0), 3)
    #     frame = cv2.line(frame, (x + w, y + h), (x, y + h), (255, 0, 0), 3)
    #     frame = cv2.line(frame, (x + w, y + h), (x + w, y), (255, 0, 0), 3)


    cv2.imshow('face', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        # break out of the while loop
        c = False
        break

    ret, frame = vid.read()






