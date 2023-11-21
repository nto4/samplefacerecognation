import cv2
import face_recognition

# Yüzleri saklamak için bir sözlük (ID: yüz konumu)
known_faces = {}

video_path = '20231121081756.mp4'
video_capture = cv2.VideoCapture(video_path)

frame_number = 0
process_every_n_frames = 5  # Her 30 karede bir yüz tanıma yapılacak

face_id_counter = 0  # Her yüze eşsiz bir ID atamak için sayaç

while video_capture.isOpened():
    ret, frame = video_capture.read()
    frame =  cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if not ret:
        break

    frame_number += 1

    if frame_number % process_every_n_frames == 0:  # Her process_every_n_frames'de bir yüz tanıma yap
        # Convert the frame from BGR to RGB (required by face_recognition)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Yüzü daha önce tanımladıysak
            found = False
            for face_id, saved_face_encoding in known_faces.items():
                # Kaydedilmiş yüzle karşılaştırma yap
                match = face_recognition.compare_faces([saved_face_encoding], face_encoding)
                if match[0]:
                    found = True
                    # Eşleşen yüz bulunduğunda ID'sini göster
                    cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {face_id}", (face_location[3], face_location[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                    print(f"Face Recognaized - ID: {face_id +1 }")
                    break

            # Yeni bir yüz tanımlandıysa
            if not found:
                face_id_counter += 1
                known_faces[face_id_counter] = face_encoding

                cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), (0, 0, 255), 2)
                cv2.putText(frame, f"ID: {face_id_counter}", (face_location[3], face_location[2] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Added new face - ID: {face_id_counter}")

    cv2.imshow('Video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
