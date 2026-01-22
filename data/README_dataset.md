# Dataset folder

This project ships with a small **synthetic** dataset (`courses.csv`, `ratings.csv`) so you can run everything end-to-end.

If your Coursera lab provides an official dataset, you can replace these files with the real ones as long as you keep the columns:

- `courses.csv`: `course_id`, `title`, `genres` (pipe-separated), optional `enrollment`
- `ratings.csv`: `user_id`, `course_id`, `rating` (1–5)

After replacing, rerun:
- `python src/eda.py`
- `python src/evaluate.py`
