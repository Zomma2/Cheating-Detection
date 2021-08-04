# Cheating-Detection
![Cheating Detection](http://www.takepart.com/sites/default/files/styles/large/public/CheatingHandAtlantic.jpg)
The supervision of students in an exam is a manual time-consuming job that can be automated completely by using various machine learning algorithms in this section ML/DL techniques used in cheating detection process is highlighted and identified.
----
Creation of automated exam supervision require multiple parts to be figured out to mimic the real/original process of supervision:
1. The student should be supervised not to have any tool to cheat with, the cheating objects can be divided into:
• Physical cheating tools like (Books, Mobile, screens, etc.)
o For this purpose, object detection algorithm should be made to detect if a student
is. Using any physical objects to cheat
• Voice-based cheating tools like (Someone in the room trying to help the student or any
voice-based material that might be considered as a cheating tool)
o For this purpose, a sound classifier must be able to detect if any student if cheating
using voice-based ways.
-----
2. The student should not be able to look around him but only to look at his exam
o For this purpose, a face pose estimator must be implemented to detect if student is not
looking straight into his exam for long periods of time
