import re
from bs4 import BeautifulSoup


def clean_string(line: str) -> str:
    # replace white space
    return line.replace("\u3000", " ").replace("\u200b", " ").replace("u3000", " ").replace("u200b", " ")


def extract_recursively(element) -> str:
    newline_tag_list = ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"]
    seperate_tag_list = []
    skip_tag_list = ["script", "style", "header", "footer", "<address>"]

    if element.name is None:
        line_text = clean_string(element.text)
        return "" if len(line_text.strip()) == 0 else line_text.strip()
    
    # elif element.attrs.get('style') is not None and re.search(r'display\s*:\s*none|visibility\s*:\s*hidden', element.attrs.get('style')) is not None:
    #     print(element.text)
    #     text = element.text
    #     return ""
    
    elif element.name in skip_tag_list:
        return ""
    
    elif element.name == "table":
        # Table
        table_string = "\n".join([str(c) for c in element.contents])
        table_string = """<table>\n{}\n</table>""".format(clean_string(table_string).strip())
        return table_string + "\n"

    elif element.name == "img":
        if element.get("src") and len(element["src"].strip()) > 0:
            if element.get("alt"):
                # Formula
                formula_text = element["alt"].strip()
                if formula_text.endswith("\\\\"):
                    line_text = formula_text
                    line_text = line_text + "\n"
                    return clean_string(line_text)
                else:
                    line_text = formula_text
                    return clean_string(line_text)
            elif element.get("class") and "lazy" not in element["class"]:
                # imgage hyperlink
                line_text = element["src"].strip()
                return clean_string(line_text) + "\n"
            else:
                return ""
        else:
            return ""
            
    elif element.name in newline_tag_list:
        return "".join([extract_recursively(c) for c in element.contents]) + "\n"
    elif element.name in seperate_tag_list:
        return "".join([extract_recursively(c) for c in element.contents]) + " "
    else:
        return "".join([extract_recursively(c) for c in element.contents])


def parse_html_text(html_content: str) -> str:
    """
    :param html_content: string, the raw html page text
    :return: string, the extracted valuable text
    """
    # wrapped_html_content = '<div class="div_Wrapper">' + html_content + "</div>"
    # soup = BeautifulSoup(wrapped_html_content, 'html.parser')
    
    # main_article = soup.find_all('div', attrs={'class': 'div_Wrapper'})

    soup = BeautifulSoup(html_content, 'html.parser')
    main_article = soup.find_all('body')

    page_text = extract_recursively(main_article[0])
    page_text = re.sub("\n{2,}", "\n\n", page_text)

    return page_text.strip()


def parse_html_title(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    
    title_elements = soup.find_all("title")

    if len(title_elements) > 0:
        title = title_elements[0].text
    else:
        title = ""
    return title.strip()