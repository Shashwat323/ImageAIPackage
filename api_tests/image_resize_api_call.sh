curl -X POST http://127.0.0.1:5000/api/resize \
  -F "image=@TestImg1.jpg" \
  -F "crop=false" \
  -F "new_width=300" \
  -F "output_path=api_tests/"
read -p "Press Enter to exit"