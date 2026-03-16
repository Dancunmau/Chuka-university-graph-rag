# Evaluation Test Queries: Chuka University GraphRAG

These queries are used to evaluate system correctness, robustness, and reliability.

##  Official Proposal Queries (Appendix C)
These 10 queries are the primary evaluation criteria defined in the project proposal.

1. **What units do I take this semester?**
   - *Target Path*: Graph (Programme -> Unit traversal)
   - *Expected Accuracy*: 100%
2. **Show my timetable for Monday**
   - *Target Path*: Graph (Unit -> TimetableSlot traversal)
   - *Expected Accuracy*: 100%
3. **Find past papers for COSC 482**
   - *Target Path*: Graph (Unit -> PastPaper link)
   - *Expected Accuracy*: 100%
4. **How much do I pay per semester?**
   - *Target Path*: Graph (Programme fee property)
   - *Expected Accuracy*: 100%
5. **What's the exam policy for missed exams?**
   - *Target Path*: Semantic (Handbook/Policies search)
   - *Expected Accuracy*: >80%
6. **When is the COSC 482 exam?**
   - *Target Path*: Graph (Unit attributes)
   - *Expected Accuracy*: 100%
7. **What rooms is my class in?**
   - *Target Path*: Graph (TimetableSlot room property)
   - *Expected Accuracy*: 100%
8. **How do I defer my studies?**
   - *Target Path*: Semantic (Handbook procedure)
   - *Expected Accuracy*: >80%
9. **What happens if I fail a course?**
   - *Target Path*: Semantic (Academic regulations)
   - *Expected Accuracy*: >80%
10. **Show me units for Year 3 Semester 2**
    - *Target Path*: Graph (Programme -> Unit filter)
    - *Expected Accuracy*: 100%

---

## Extended Test Suite (50 Queries)
The following queries cover a broader spectrum of student needs for stress testing the Hybrid retrieval paths.

## Fees & Financials
1. What are the total fees for a BSc in Computer Science?
2. How do I pay my tuition fees?
3. What is the cost of a Master's program?
4. How much does a PhD in Education cost per semester?
5. Are there any scholarships available for needy students?
6. When is the deadline for fee payment?
7. Can I pay my fees in installments?
8. What happens if I have a fee balance during exams?
9. How much is the caution money?
10. Where can I find the official fee structure document?

## Units & Academic Sequence
11. What units am I taking in Computer Science Year 1 Semester 1?
12. List all the units for BEd Arts Year 2.
13. What units are currently offered in the Jan-April 2026 semester?
14. Which department offers the COSC 121 unit?
15. What are the units for Nursing Year 4?
16. Give me a list of all programmes in the Faculty of Science.
17. What do I study in the Bachelor of Commerce program?
18. Are there any common units for all first-year students?
19. Which units are prerequisite for Advanced Programming?
20. Show me the course units for Law Year 3 Semester 2.

## Timetables & Scheduling
21. When is my COSC 121 class scheduled?
22. Where can I find the semester teaching timetable?
23. What time is the unit for Software Engineering?
24. Which units are taught in the Graduation Square?
25. Is there a class for BEd Science on Monday mornings?
26. Who is the lecturer for Database Systems this semester?
27. When is the exam for Discrete Mathematics?
28. Which venue is being used for COSC 111?
29. Does the department have a specialized computer lab timetable?
30. Where is the Jan-April 2026 timetable?

## Past Papers
31. Do you have past papers for COSC 121?
32. Link me to the 2023 past paper for Statistics.
33. I need past papers for all units in Computer Science Year 2.
34. Search for 'Operating Systems' past papers.
35. How can I download past papers from the repository?
36. Are there 2024 past papers available?
37. Find papers for 'Object Oriented Programming'.
38. What is the most recent paper for Data Structures?
39. Can I access papers by unit code only?
40. Give me all past papers from the Department of Humanities.

## Policies & Regulations
41. What is the minimum GPA for graduation?
42. How do I apply for a course transfer?
43. What is the policy on missing an exam?
44. How many units can I retake?
45. What are the library opening hours?
46. How do I join the student union?
47. What happens if I am caught cheating in an exam?
48. How do I request an academic transcript?
49. What is the procedure for semester deferment?
50. Where can I find the student handbook online?
