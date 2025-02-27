import cv2
import numpy as np
import os
import datetime
from PIL import Image
import csv
import time

def display_menu():

    print("\n===== FACE RECOGNITION ATTENDANCE SYSTEM =====")
    print("1. Add New Student")
    print("2. Train Recognition System")
    print("3. Mark Attendance")
    print("4. View Attendance Records")
    print("5. Exit")
    return input("Enter your choice (1-5): ")

def create_user(f_id, name):

    f_dir = 'dataset'
    if not os.path.isdir(f_dir):
        os.mkdir(f_dir)
        
    f_name = name
    path = os.path.join(f_dir, f_name)  
    if not os.path.isdir(path):
        os.mkdir(path)
    

    web = cv2.VideoCapture(0)
    web.set(3, 640)
    web.set(4, 480) 


    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"\nCapturing face data for {name} (ID: {f_id})...")
    print("Press ESC to cancel or wait for 30 images to be captured.")
    
    counter = 0
    while True:
        ret, img = web.read()
        if not ret:
            print("Failed to grab frame from camera. Check if camera is connected!")
            break
            
        img = cv2.flip(img, 1)  
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        multi_face = faces.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in multi_face: 
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            counter += 1
            
            cv2.imwrite(f"{path}/{name}_{f_id}_{counter}.jpg", gray[y:y+h, x:x+w])
            

            cv2.putText(img, f"Images Captured: {counter}/30", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Capturing Face", img)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break
        elif counter >= 30:
            break

    web.release()
    cv2.destroyAllWindows()
    
    if counter > 0:
        print(f"\nSuccessfully captured {counter} images for {name}.")
        return True
    else:
        print("\nNo face detected. Please try again.")
        return False

def train_data():

    print("\nTraining the recognition system...")
    
    database = 'dataset'
    if not os.path.isdir(database):
        print("Dataset directory not found. Please add students first.")
        return 0
        
    img_dir = [x[0] for x in os.walk(database)][1:]
    
    if not img_dir:
        print("No student data found. Please add students first.")
        return 0
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face_samples = []
    face_ids = []

    for path in img_dir:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

        for imagePath in imagePaths:
            try:
                filename = os.path.splitext(os.path.basename(imagePath))[0]
                face_id = int(filename.split("_")[1])  # Extract ID
                
                PIL_img = Image.open(imagePath).convert('L')
                img_numpy = np.array(PIL_img, 'uint8')

                faces = detector.detectMultiScale(img_numpy, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in faces:
                    face_samples.append(img_numpy[y:y+h, x:x+w])
                    face_ids.append(face_id)
            except Exception as e:
                print(f"Error processing {imagePath}: {e}")
                continue

    if len(face_samples) == 0:
        print("Error: No valid face samples found.")
        return 0

    recognizer.train(face_samples, np.array(face_ids))
    
    if not os.path.isdir('trainer'):
        os.mkdir('trainer')
    recognizer.write('trainer/trainer.yml')
    
    unique_faces = len(np.unique(face_ids))
    print(f'\nTraining complete! {unique_faces} faces trained.')
    return unique_faces

def load_student_data():

    students = {}
    if not os.path.isfile('students.csv'):

        with open('students.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['ID', 'Name'])
        return students
    
    with open('students.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row) >= 2:
                students[int(row[0])] = row[1]
    return students

def save_student_data(students):

    with open('students.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Name'])
        for student_id, name in students.items():
            writer.writerow([student_id, name])

def mark_attendance(students):

    if not students:
        print("No students registered. Please add students first.")
        return

    if not os.path.isfile('trainer/trainer.yml'):
        print("Recognition model not found. Please train the system first.")
        return

    attendance_dir = 'attendance'
    if not os.path.isdir(attendance_dir):
        os.mkdir(attendance_dir)

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    attendance_file = f"{attendance_dir}/attendance_{today}.csv"

    attendance_record = {}
    if os.path.isfile(attendance_file):
        with open(attendance_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    attendance_record[int(row[0])] = (row[1], row[2])
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    font = cv2.FONT_HERSHEY_SIMPLEX
    recognized_students = set()
    
    print("\nStarting face recognition for attendance...")
    print("Press ESC to stop the recognition process.")
    
    vid = cv2.VideoCapture(0)
    vid.set(3, 640)
    vid.set(4, 480)

    minW = 0.1 * vid.get(3)
    minH = 0.1 * vid.get(4)
    
    recognition_counters = {}

    while True:
        ret, img = vid.read()
        if not ret:
            print("Failed to grab frame from camera. Check if camera is connected!")
            break
            
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                if id in students:
                    person_name = students[id]
                    confidence_text = f"{round(100-confidence)}%"

                    if id not in recognition_counters:
                        recognition_counters[id] = 0

                    recognition_counters[id] += 1

                    if recognition_counters[id] >= 10 and id not in recognized_students:
                        recognized_students.add(id)
                        current_time = datetime.datetime.now().strftime("%H:%M:%S")
                        attendance_record[id] = (person_name, current_time)
                        print(f"✅ Attendance marked for {person_name} (ID: {id})")
                else:
                    person_name = "Unknown ID"
                    confidence_text = f"{round(100-confidence)}%"
            else:
                person_name = "Unknown"
                confidence_text = f"{round(100-confidence)}%"

            cv2.putText(img, str(person_name), (x+5, y-5), font, 1, (0, 255, 0), 2)
            cv2.putText(img, confidence_text, (x+5, y+h-5), font, 0.7, (255, 255, 0), 1)

        cv2.putText(img, f"Students Present: {len(recognized_students)}", (10, 30), 
                   font, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Attendance System", img)
        
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break

    vid.release()
    cv2.destroyAllWindows()
    
    with open(attendance_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID', 'Name', 'Time'])
        for student_id, (name, time) in attendance_record.items():
            writer.writerow([student_id, name, time])
    
    print(f"\nAttendance recorded for {len(recognized_students)} students.")
    print(f"Attendance saved to {attendance_file}")

def view_attendance():

    attendance_dir = 'attendance'
    if not os.path.isdir(attendance_dir):
        print("No attendance records found.")
        return
    
    attendance_files = [f for f in os.listdir(attendance_dir) if f.startswith("attendance_") and f.endswith(".csv")]
    
    if not attendance_files:
        print("No attendance records found.")
        return

    attendance_files.sort(reverse=True)
    
    print("\n===== AVAILABLE ATTENDANCE RECORDS =====")
    for i, file in enumerate(attendance_files, 1):
        date = file.replace("attendance_", "").replace(".csv", "")
        print(f"{i}. {date}")
    
    try:
        choice = int(input("\nEnter record number to view (0 to cancel): "))
        if choice == 0:
            return
        
        if 1 <= choice <= len(attendance_files):
            selected_file = os.path.join(attendance_dir, attendance_files[choice-1])
            date = attendance_files[choice-1].replace("attendance_", "").replace(".csv", "")
            
            print(f"\n===== ATTENDANCE RECORD FOR {date} =====")
            print(f"{'ID':<5} {'Name':<20} {'Time':<10}")
            print("-" * 35)
            
            with open(selected_file, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 3:
                        print(f"{row[0]:<5} {row[1]:<20} {row[2]:<10}")
        else:
            print("Invalid choice.")
    except ValueError:
        print("Invalid input. Please enter a number.")

def add_new_student():

    students = load_student_data()

    next_id = 1
    if students:
        next_id = max(students.keys()) + 1
    
    name = input("\nEnter student name: ")
    if not name:
        print("Name cannot be empty. Aborting.")
        return

    success = create_user(next_id, name)
    if success:

        students[next_id] = name
        save_student_data(students)
        print(f"\nStudent {name} added successfully with ID {next_id}.")

        choice = input("Do you want to train the recognition system now? (y/n): ")
        if choice.lower() == 'y':
            train_data()
    else:
        print("\nFailed to add student. Please try again.")

def main():

    for directory in ['dataset', 'trainer', 'attendance']:
        if not os.path.isdir(directory):
            os.mkdir(directory)
    
    while True:
        choice = display_menu()
        
        if choice == '1':
            add_new_student()
        elif choice == '2':
            train_data()
        elif choice == '3':
            students = load_student_data()
            mark_attendance(students)
        elif choice == '4':
            view_attendance()
        elif choice == '5':
            print("\nExiting system. Goodbye!")
            break
        else:
            print("\nInvalid choice. Please try again.")
        
        time.sleep(1)

if __name__ == "__main__":
    main()