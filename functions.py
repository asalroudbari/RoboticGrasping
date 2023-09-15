import cv2
import numpy as np
import math
import torch 
from ultralytics import YOLO
from PIL import Image
import client
import serial
import sys
import time


def edge_detection(im, im2, label):

    edges = cv2.Canny(im,10,70)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Create a copy of the original image
    image_copy = im2
    flag = 0
    # Store valid bounding boxes
    boxes = []
    angles = []
    heights = []
    widths = []
    x_grasp = []
    y_grasp = []

    for ctr in contours:
        # Obtain minimum area rectangle
        rect = cv2.minAreaRect(ctr)

        # Get the four corners of the rectangle
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Calculate the width and height of the bounding box
        width = np.linalg.norm(box[0] - box[1])
        height = np.linalg.norm(box[1] - box[2])
        rows, cols = image_copy.shape[:2]
        # Specify the minimum size threshold (modify as desired)
        min_size = 30

        # Check if the width or height is smaller than the minimum size threshold
        if width < min_size or height < min_size:
            continue  # Skip the small bounding box
        if label == "metal box": 
            area = width* height
            if area < 200000:
                flag = -1
            else:
                cx = box[0][0] + box[1][0]
                cy = box[0][1] + box[1][1]
                cx , cy = int(0.5*cx) , int(0.5*cy)
                cy+= 20
                width = 40
        if flag == -1:
            break
        # Draw the bounding box on the image
        cv2.drawContours(image_copy, [box], 0, (0, 255, 255), 2)

        # Get the centroid of the bounding box
        centroid = rect[0]

        #draw a line on the box
        vx, vy, x, y = cv2.fitLine(ctr, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the orientation angle of the bounding box
        slope = -vy/vx
        angle = math.degrees(math.atan(slope))
        angle = round(angle , 2)
        angle_degrees = angle
        angles.append(angle)

        # Draw the 'X' point on the line
        mapped_x = int(centroid[0])
        mapped_y = int(centroid[1])

        # Calculate the y-coordinate of the centroid of the bounding box
        cy = mapped_y

        # Calculate the slope and y-intercept of the fitted line
        if vx != 0:
            m = vy / vx
            b = y - m * x
        else:
            m = 0
            b = x
        # Calculate the corresponding x-coordinate on the line for the y-coordinate of the centroid of the bounding box
        if m != 0:
            cx = int((cy - b) / m)
        else:
            cx = int(x)
        # Write the orientation angle near the bounding box
        cv2.putText(image_copy, str(angle_degrees), (int(centroid[0] + 17), int(centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if label == "tray":
            cx = int(centroid[0])
            cy = int(centroid[1])
            # Draw a marker or a line from the centroid of the bounding box to the corresponding point on the line
        cv2.drawMarker(image_copy, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_STAR, thickness=2)

            # Append the box coordinates, height, and width to boxes
        width = round(width, 2)
        boxes.append(box)
        heights.append(height)
        widths.append(width)
        x_grasp.append(cx)
        y_grasp.append(cy)

    # Return the resulting image, boxes, heights, and angles
    return image_copy, boxes, heights, angles, widths, x_grasp , y_grasp, flag


def box_out(results, img, classNames):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            # class name
            cls = int(box.cls[0])
            print("Class name, confidence -->", classNames[cls] + ",", str(confidence))
            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls]+ ",conf:"+ str(confidence), org, font, fontScale, color, thickness)
            return img 


def print_results (label, orientation, width, cox, coy, flag):
    if flag == 0:
        print(label)
        print("Gripper:  "+ "orientation:"+ str(orientation)+ ",     width:"+ str(width))
        print("Delta:  "+"X:"+ str(cox)+ ",     Y:"+ str(coy))
        print("-------------------------------------------")
         

def tes_generator(masks, i):
    maski= masks[i]
    mask = maski.data[0].numpy()
    mask_img = Image.fromarray(mask, 'I')
    mask_img = mask_img.convert('RGB')
    mask_img.save("masked.jpg")

def grasp_point_estimation(img):
    model = YOLO('yolox.pt')
    labels = []
    mid_x = []
    mid_y = []
    orientations = []
    results = model.predict(img, conf= 0.6)
    classNames = ["Can", "bottle", "layed bottle", "layed can","metal box", "metal fork","metal knife",  "metal spoon", "pfork", "pglass", "pknife", "pspoon", "salt shaker", "sause", "square plate", "tray"]
    result = results[0]
    masks = result.masks
    boxes = result.boxes
    masknum = len(masks)
    for i in range (masknum):
        cls = int(boxes[i].cls[0])
        label = classNames[cls]
        copied = img.copy()
        tes_generator(masks, i)
        test_im = cv2.imread('masked.jpg')
            #outputs of the edge detectioin function 
        output, bbox, height, orientation, width, cox , coy, flag = edge_detection(test_im, copied, label)
        if not label == "tray":
                # Save output for check
            cv2.imwrite('output' + str(i)+ '.jpg', output)
                #printing necessary data for grasping
            print_results(label, orientation, width, cox, coy, flag)
        else:
            print("Tray Detected:")
            print("location tray:", cox[0], ", ", coy[0],"    orientation:", orientation[0] )
            cv2.imwrite('output' + str(i)+ '.jpg', output)
        labels.append(label)
        mid_x.append(cox)
        mid_y.append(coy)
        orientations.append(orientation)
    return labels, mid_x, mid_y, orientations


def tray_detection(img):
    labels, mid_x, mid_y, orientation= grasp_point_estimation(img)
    other_labels = []
    other_x = []
    other_y = []
    other_orient= []
    i=0
    for label in labels:
        if label == "tray":
            x_tray = mid_x[i]
            y_tray = mid_y[i]
            angle = orientation[i]
            i+=1
        else: 
            other_labels.append(label)
            other_x.append(mid_x[i])
            other_y.append(mid_y[i])
            other_orient.append(orientation[i])
            i+=1
    return x_tray, y_tray, angle, other_labels, other_x, other_y, other_orient

def arrange_tray(img):
    x_tray, y_tray, angle_tray, labels, mid_x, mid_y, orientation = tray_detection(img)
    x_tray, y_tray = x_tray[0], y_tray[0]
    arrange_label = []
    r = []
    theta = []
    position_angles = []
    i=0
    angle_tray= angle_tray[0]
    for label in labels:
            horiz = "left"
            vert = "top"
            arrange_label.append(label)
            x, y = mid_x[i][0], mid_y[i][0]
            if x > x_tray:
                horiz = "right"
            if y > y_tray:
                vert = "bottom"
            tray_loc = np.array((x_tray, y_tray, 0))
            point = np.array((x, y, 0))
            radius = np.linalg.norm(point - tray_loc)
            r.append(round(radius,3))
            position_angle = np.arctan((y_tray - y)/(x-x_tray))
            position_angle = math.degrees(position_angle)
            if vert == "bottom" and position_angle > 0: 
                position_angle-= 180
            elif horiz == "left" and position_angle < 0:
                position_angle+= 180

            position_angle = round(position_angle,2)
            position_angles.append(position_angle)
            print("position angle:  ", position_angle)
            dtheta = position_angle - angle_tray
            theta.append(dtheta)
            print(label + "  should be placed " + vert+ " "+ horiz)
            i+=1
    print("-------------------------------------------")
    return arrange_label, r, theta, position_angles


def place2(scene, input_image):
    scene_copy = scene
    x_tray, y_tray, angle_tray, other_labels, other_x, other_y, other_orient = tray_detection(scene)
    x_tray, y_tray, angle_tray= x_tray[0], y_tray[0], angle_tray[0]
    x_tray2, y_tray2, angle_tray2, other_labels2, other_x2, other_y2, other_orient2 = tray_detection(input_image)
    final_labels = []
    final_coordinates = []
    final_angles = []
    num_mask = len(other_labels)
    for i in range (num_mask):
        final_labels.append(other_labels[i])
        init_x = other_x[i][0]
        init_y = other_y[i][0]
        init_co = [init_x, init_y]
        final_coordinates.append(init_co)
        init_angle = angle_tray
        final_angles.append(init_angle)
        final_labels.append(other_labels2[i])
        final_x = other_x2[i][0]
        final_y = other_y2[i][0]
        final_co = [final_x, final_y]
        final_coordinates.append(final_co)
        final_angle = angle_tray
        final_angles.append(final_angle)
        cv2.drawMarker(scene_copy, (final_x, final_y), (0, 0, 255), markerType=cv2.MARKER_STAR, thickness=2)
        cv2.imwrite('final_'+ str(i)+'.jpg', scene_copy)
    return final_labels, final_coordinates, final_angles
        
        

    
def place_tray(scene, input_image):
    scene_copy = scene
    x_tray, y_tray, angle_tray, other_labels, other_x, other_y, other_orient = tray_detection(scene)
    x_tray, y_tray, angle_tray= x_tray[0], y_tray[0], angle_tray[0]
    arrange_label, r, theta, position_angles = arrange_tray(input_image)
    num_mask = len(other_labels)
    num_label = len(arrange_label)
    final_labels = []
    final_coordinates = []
    final_angles = []

    for i in range (num_mask):
        desired_label = arrange_label[i]
        for j in range(num_label):
            if other_labels[j] ==desired_label:
                #picking process
                final_labels.append(desired_label)
                init_x = other_x[i][0]
                init_y = other_y[i][0]
                init_co = [init_x, init_y]
                final_coordinates.append(init_co)
                init_angle = other_orient[i][0]
                final_angles.append(init_angle)
                # placing process
                final_labels.append(desired_label)
                position_place = position_angles[j] + angle_tray
                position_place = math.radians(position_place)
                radius = r[j]
                x_place = x_tray + np.cos(position_place)*radius
                y_place = y_tray + np.sin(position_place)*radius
                x_place, y_place = int(x_place), int(y_place)
                coordinate = [x_place, y_place]
                final_coordinates.append(coordinate)
                theta_place = angle_tray
                final_angles.append(theta_place)
                cv2.drawMarker(scene_copy, (x_place, y_place), (0, 0, 255), markerType=cv2.MARKER_STAR, thickness=2)
                cv2.imwrite('final_'+ str(j)+'.jpg', scene_copy)
            
    return final_labels, final_coordinates, final_angles




def graspObjects(objects):
    ND = 1
    whose_hand = input("Navid or Daalvand? N/D") 
    if whose_hand.upper() == "N":
        ND = 0 #navid hand
    elif whose_hand.upper() == "D":
        ND = 1 #daalvand hand
    names = objects[0]
    centers = objects[1]
    angles = objects[2]
    curr_angle = 0
    closing = 1300
    is_placing = False #start picking
    ser = serial.Serial('COM3',38400, timeout=None)
    time.sleep(10)
    print('Grasping starts now!')
    if ND == 0:
        ser.write(b"H1\r\n")
    elif ND==1: 
        ser.write(b"h\r\n")
    while 1:
        result = ser.readline().decode("utf-8")
        print(f"Gripper: {result}")
        if result[0:4] == "Done":
            break
    client.order("command", "forward")
    robot_capturing_coord = [client.Result[1],client.Result[2],client.Result[3]]
    for i in range(0,len(names)):
        z_obj = 2 
        # flag = 0 : pick ,,, flag = 1 place 
        if i % 2 == 0 :
            is_placing = False
        elif i % 2 == 1:
            is_placing = True
        if names[i] == "Can":
            z_obj = 8
            closing = 650
        elif names[i] == "layed bottle":
            z_obj = 1.5
        elif names[i] == "pfork":
            z_obj = 0.7
        elif names[i] == "pglass":
            z_obj = 7.5
        # calculate x,y,z
        [x_w,y_w,z_w] = calculate_XYZ(centers[i][0], centers[i][1],z_obj,robot_capturing_coord, ND)
        # starts moving 
        # point 0
        client.order("move", f"{x_w},{y_w},{z_w+12}")
        #goes to the top of the object, rotates
        # if ND == 0: 
        #     ser.write(f"H0G0R{angles[i]-curr_angle}".encode("utf-8"))
        # elif ND == 1:
        #     ser.write(f"h".encode("utf-8"))
        curr_angle = angles[i]
        # while 1:
        #     result = ser.readline().decode("utf-8")
        #     print(f"Gripper: {result}")
        #     if result[0:4] == "Done":
        #         break
        if client.Result == "success":
            if not is_placing:
                if ND == 0: 
                    ser.write(b"H1\r\n") #Just to be sure gripper is at its home state
                elif ND == 1:
                    ser.write(b"h") 
            # while 1:
            #     result = ser.readline().decode("utf-8")
            #     print(f"Gripper: {result}")
            #     if result[0:4] == "Done":
            #         break      navid
        if client.Result == "success":
            if is_placing: # to check wether we rich the tray 
                z_w += 1.5
            #point 1
            client.order("move", f"{x_w},{y_w},{z_w-1.75}")
        if client.Result == "success":
            if not is_placing:
                if ND == 0:
                    ser.write(f"H1G{closing}R0".encode("utf-8"))
                elif ND == 1: 
                    ser.write(f"g".encode("utf-8"))
            elif is_placing:
                if ND == 0:
                    ser.write(f"H1".encode("utf-8"))
                elif ND == 1: 
                    ser.write(f"h".encode("utf-8"))

            while 1:
                result = ser.readline().decode("utf-8")
                print(f"Gripper: {result}")
                if result[0:4] == "Done":
                    break
        client.order("move", f"{x_w},{y_w},{z_w+10}")
        if client.Result == "success":
            if is_placing:
                if ND == 0:
                    ser.write(b"H1\r\n")
                elif ND == 1: 
                    ser.write(b"h\r\n")
            

        
            while 1:
                print('grabbed')
                result = ser.readline().decode("utf-8")
                print(f"Gripper: {result}")
                if result[0:4] == "Done":
                    break
            

def calculate_XYZ(u,v , z_obj, robot_capturing_coord, ND):
    robot_capturing_coord_default = np.array([0,0,-37])
    Values_tr00,Values_tr01,Values_tr10,Values_tr11,Values_off0,Values_off1 = np.load('values.npy')

    H = 50 - z_obj + 37 + robot_capturing_coord[2]
    print('hight cam from obj = ',H)

    p00,p01,p10,p11 = np.poly1d(Values_tr00),np.poly1d(Values_tr01),np.poly1d(Values_tr10),np.poly1d(Values_tr11)
    tr_Hight = np.array([[p00(H),p01(H),0],[p10(H),p11(H),0],[0,0,0]])
    offp0,offp1 = np.poly1d(Values_off0), np.poly1d(Values_off1)
    offset_Hight = np.array([offp0(H),offp1(H),-64.5 +50 -H]) + (robot_capturing_coord - robot_capturing_coord_default) + [-1.7,-0.3,0]

    regenerated_output_centered = np.dot([u,v,0],tr_Hight) + offset_Hight - ND*[0,0,6.5]

    return regenerated_output_centered

