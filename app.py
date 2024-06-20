import os
from flask import Flask, request, render_template_string
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download the VADER lexicon
nltk.download('vader_lexicon')

app = Flask(__name__)

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# HTML template for the form
form_html = """
<!doctype html>
<title>Bearblog Scraper</title>
<style>
    body {
        place-items: center;
        align-items: center;
        text-align: left;
        padding: 20px;
    }
    h1 {
        text-align: center;
    }
    .container {
        border: 1px solid #ccc;
        padding: 20px;
        margin-top: 20px;
    }
</style>
<h1>Bearblog Sentiment Scores</h1>
<body>
<p>Find a bearblog you like <a href="https://bearblog.dev/discover">here</a>. Copy and paste the domain below.</p>
<form action="/" method="post">
    <b>Bearblog Domain:</b> <input type="text" name="domain">
    <input type="submit" value="Do the thing">
</form>
<div class="container">
    <p>You'll receive the sentiment analysis results for each post under that domain.</p>
    <p>Sentiment scores:</p>
    <i>Example scores:</i> <code>{'neg': 0.046, 'neu': 0.82, 'pos': 0.134, 'compound': 0.9851}</code>
    <ul>
        <li>Negative (neg): This score indicates the proportion of the text that is perceived as negative. Range: [0, 1]. Example: If neg is 0.046, it means 4.6% of the text is negative.</li>
        <li>Neutral (neu): This score indicates the proportion of the text that is perceived as neutral. Range: [0, 1]. Example: If neu is 0.82, it means 82% of the text is neutral.</li>
        <li>Positive (pos): This score indicates the proportion of the text that is perceived as positive. Range: [0, 1]. Example: If pos is 0.134, it means 13.4% of the text is positive.</li>
        <li>Compound (compound): This score is a normalized, weighted composite score that calculates the overall sentiment of the text. It combines the values of neg, neu, and pos scores. Range: [-1, 1]. Interpretation: A score close to 1 indicates extremely positive sentiment. A score close to -1 indicates extremely negative sentiment. A score around 0 indicates neutral sentiment.</li>
    </ul>
</div>
{% if file_content %}
<div class="container">
    <h2>Sentiment Analysis Results</h2>
    <pre style="max-height: 600px; overflow-y: scroll; white-space: pre-wrap;">{{ file_content }}</pre>
</div>
{% endif %}
</body>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        domain = request.form['domain']
        if not domain.startswith('http'):
            domain = 'http://' + domain
        return scrape_and_analyze(domain)
    return render_template_string(form_html)

def scrape_and_analyze(domain):
    try:
        # Define the base URL and blog list URL
        base_url = domain
        blog_list_url = urljoin(base_url, 'blog/')
        
        # Send a GET request to the blog list URL
        response = requests.get(blog_list_url)
        
        # Parse the content of the response with Beautiful Soup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the section containing blog posts
        blog_posts = soup.find('ul', class_='blog-posts')
        
        # If blog posts section is not found, raise an error
        if blog_posts is None:
            raise ValueError("Could not find blog posts section")

        # Extracting each blog post entry
        blogs = []
        for li in blog_posts.find_all('li'):
            title = li.find('a').text
            link = li.find('a')['href']
            date = li.find('span').text if li.find('span') else "No date provided"
            blogs.append({'title': title, 'link': link, 'date': date})

        # Perform sentiment analysis on each blog content and save results
        output_file = 'sentiment_analysis.txt'
        compound_scores = []  # List to store all compound scores
        
        with open(output_file, 'w') as f:
            for blog in blogs:
                # Construct the full URL for the blog post
                blog_url = urljoin(base_url, blog['link'])
                blog_response = requests.get(blog_url)
                
                # Check if the request was successful
                if blog_response.status_code == 200:
                    blog_soup = BeautifulSoup(blog_response.content, 'html.parser')
                    
                    # Find the main element containing the blog content
                    main_content = blog_soup.find('main')
                    if main_content:
                        content = []
                        # Extract text from h1, h2, h3, and p elements
                        for tag in main_content.find_all(['h1', 'h2', 'h3', 'p']):
                            content.append(tag.get_text(strip=True))
                        blog_content = "\n".join(content)
                    else:
                        blog_content = "Content not found"
                else:
                    blog_content = "Error: Unable to fetch content (status code: {})".format(blog_response.status_code)

                # Run sentiment analysis
                sentiment_scores = sid.polarity_scores(blog_content)
                compound_scores.append(sentiment_scores['compound'])  # Add the score to the list

                # Write the results to the output file
                f.write(f"Title: {blog['title']}\n")
                f.write(f"Link: {blog_url}\n")
                f.write(f"Date: {blog['date']}\n")
                f.write(f"Content:\n{blog_content}\n\n")
                f.write(f"Sentiment Scores: {sentiment_scores}\n")
                f.write("="*80 + "\n\n")

        # Calculate the average of all compound scores
        average_compound_score = sum(compound_scores) / len(compound_scores) if compound_scores else 0

        # Insert the average compound score at the beginning of the file
        with open(output_file, 'r') as f:
            file_content = f.read()
        with open(output_file, 'w') as f:
            f.write(f"Average Compound Score: {average_compound_score}\n\n")
            f.write(file_content)

        # Read the file content for display
        with open(output_file, 'r') as f:
            file_content = f.read()

        return render_template_string(form_html, file_content=file_content)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)