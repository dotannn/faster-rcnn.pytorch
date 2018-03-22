import cv2



def draw_bbox_cross(frame, inst):
    far_color = (100, 255, 100)
    mid_color = (128, 255, 255)
    near_color = (128, 128, 255)

    color = far_color
    if hasattr(inst, '_distance_level'):
        if inst._distance_level == 'Mid':
            color = mid_color
        elif inst._distance_level == 'Near':
            color = near_color

    height = (inst._bbox[3] - inst._bbox[1])
    width = (inst._bbox[2] - inst._bbox[0])
    thickness = 1

    px0 = (int(inst._bbox[0]), int(inst._bbox[1] + height/2.0))
    px1 = (int(inst._bbox[2]), int(inst._bbox[1] + height/2.0))

    py0 = (int(inst._bbox[0] + width/ 2.0), inst._bbox[1])
    py1 = (int(inst._bbox[0] + width/ 2.0), inst._bbox[3])

    cv2.line(frame, px0, px1,  color=color, thickness=thickness)
    cv2.line(frame, py0, py1,  color=color, thickness=thickness)

    top0 = (int(inst._bbox[0] + width/2.0 - width/10), int(inst._bbox[1]))
    top1 = (int(inst._bbox[0] + width / 2.0 + width/10), int(inst._bbox[1]))

    bottom0 = (int(inst._bbox[0] + width / 2.0 - width/10), int(inst._bbox[3]))
    bottom1 = (int(inst._bbox[0] + width / 2.0 + width/10), int(inst._bbox[3]))

    left0 = (int(inst._bbox[0]), int(inst._bbox[1] + height/2.0 - height/10.0))
    left1 = (int(inst._bbox[0]), int(inst._bbox[1] + height / 2.0 + height / 10.0))

    right0 = (int(inst._bbox[2]), int(inst._bbox[1] + height / 2.0 - height / 10.0))
    right1 = (int(inst._bbox[2]), int(inst._bbox[1] + height / 2.0 + height / 10.0))

    cv2.line(frame, top0, top1, color=color, thickness=thickness)
    cv2.line(frame, bottom0, bottom1, color=color, thickness=thickness)
    cv2.line(frame, left0, left1, color=color, thickness=thickness)
    cv2.line(frame, right0, right1, color=color, thickness=thickness)

    center = (int(inst._bbox[0] + width/2.0), int(inst._bbox[1] + height/2.0))

    name = inst._label
    fontscale = width / 70
    sz = cv2.getTextSize(name, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontscale, thickness=1)[0]

    frame = cv2.rectangle(frame, (int(center[0]-sz[0]/2.0), int(center[1]-sz[1]/2.0)),
                          (int(center[0] + sz[0] / 2.0), int(center[1] + sz[1] / 2.0)),
                          color=(0, 0, 0), thickness=-1)

    # frame = cv2.rectangle(frame, p1, p2, color=(0, 0, 255), thickness=2)
    frame = cv2.putText(frame, name, (int(center[0]-sz[0]/2.0), int(center[1]+sz[1]/2.0)),
                        fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=fontscale,
                        color=(255, 255, 255), thickness=1)


    return frame

def draw_detections(frame, insts):
    for inst in insts:
        frame = draw_bbox_cross(frame, inst)


    return frame
