# XAI Repo Guide

This is the repository for the XAI blog.

## Table of Contents

- [How to Contribute](#how-to-contribute)  
  - [Forking the Repository](#forking-the-repository)  
  - [Running the Project Locally](#running-the-project-locally)  
  - [Making Changes](#making-changes)  
- [Pull Requests](#pull-requests)  
- [Uploading Images](#uploading-images)  
- [Shortcodes](#shortcodes)  
  - [LaTeX](#latex)  
  - [Mermaid (Graphs)](#mermaid-graphs)  
  - [Other Shortcodes](#other-shortcodes)  

---

## How to Contribute

### Forking the Repository

If you **do not have** the repo on your machine:

1. Make sure you have [Git](https://git-scm.com/downloads), [Hugo](https://gohugo.io/installation/), and a code editor installed. We recommend [VSCode](https://code.visualstudio.com/), but feel free to use your favorite.

2. **Fork the repository**:  
   Go to the original repo: [https://github.com/IU-PR/xai.git](https://github.com/IU-PR/xai.git)  
   Click "Fork" in the top-right corner.

3. **Clone your forked repository (with submodules)**:
   ```bash
   git clone --recursive https://github.com/YOUR_USERNAME/xai.git
   ```

> ⚠️ If you've already cloned the repo without submodules, run:
```bash
git submodule init && git submodule update
```

---

### Running the Project Locally

To preview the site locally:

```bash
hugo server --minify --theme=hugo-book
```

You should see a line like:

```
Web Server is available at http://localhost:PORT/ (bind address 127.0.0.1)
```

Visit that address in your browser to preview your changes. The site will auto-update as you edit files.

---

### Making Changes

1. Make your changes in the appropriate `.md` or layout files.
2. Stage and commit your changes:
   ```bash
   git add .
   git commit -m "Descriptive message about what you changed"
   ```

> 🔒 These steps save and document changes **only on your local machine**. To share them, you’ll need to push and create a pull request.

---

## Pull Requests

Since you do not have direct push access to the original repository, you’ll need to submit changes via a **pull request from your fork**.

### Steps:

1. Make sure you are working on a **new branch** (not `main`):
   ```bash
   git checkout -b my-feature-branch
   ```

2. After committing your changes, push your branch:
   ```bash
   git push origin my-feature-branch
   ```

3. Go to your fork on GitHub and you should see a "Compare & pull request" button. Click it.

4. Make sure you're comparing against the original repo:  
   **base repository**: `IU-PR/xai`  
   **head repository**: `YOUR_USERNAME/xai`

5. Write a clear description of your changes and optionally suggest future improvements.

6. Click **Create pull request**.

The repo maintainers will review and merge your changes.

---

## Uploading Images

To add images to your report:

1. Create a folder with your **group name** inside the `/static/` directory.
2. Add your images to that folder:  
   Example: `/static/your_group_folder/your_image.png`

3. Reference the image in your `.md` like this:
   ```markdown
   ![alt text](/your_group_folder/your_image.png)
   ```

---

## Shortcodes

This blog template supports several shortcodes, which enable advanced formatting in Markdown.

### LaTeX

To render LaTeX formulas:

Inline formula:
```markdown
{{<katex>}}
your_latex_here
{{</katex>}}
```

Centered formula:
```markdown
{{<katex display>}}
your_centered_latex_here
{{</katex>}}
```

📘 [LaTeX Cheatsheet](https://wch.github.io/latexsheet/latexsheet.pdf)

---

### Mermaid (Graphs)

To draw diagrams using [Mermaid](https://mermaid.js.org/):

```markdown
{{<mermaid>}}
gitGraph
  commit
  commit
  branch develop
  commit
  commit
  commit
  checkout main
  commit
  commit
{{</mermaid>}}
```

There are many other diagram types—check out the [Mermaid documentation](https://mermaid.js.org/intro/).

---

### Other Shortcodes

All shortcodes use the same syntax:

```markdown
{{<shortcode>}}
...your content...
{{</shortcode>}}
```

Explore the available shortcodes in `/layouts/shortcodes`. Use them in your reports if they improve formatting or readability.
