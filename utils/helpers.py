import cv2


def draw_detections(frame, insts):
    for inst in insts:
        if inst.stability < 1:
            continue
        p1 = (int(inst._bbox[0]), int(inst._bbox[1]))
        p2 = (int(inst._bbox[2]), int(inst._bbox[3]))
        name = inst._name
        frame = cv2.rectangle(frame, p1, p2, color=(0,0,255), thickness=2 )
        text = "%s %.2f | %s (%.2fm)" % (name, inst._score, inst._distance_level, inst._distance)
        frame = cv2.putText(frame, text, (p1[0], p1[1]+30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,color=(0, 255, 0))
    return frame
