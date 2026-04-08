import logging
import re
import sys
from bs4 import BeautifulSoup
from queue import Queue
from urllib import parse, request
from queue import PriorityQueue

logging.basicConfig(level=logging.DEBUG, filename='output.log', filemode='w')
visitlog = logging.getLogger('visited')
extractlog = logging.getLogger('extracted')


def parse_links(root, html):
    soup = BeautifulSoup(html, 'html.parser')
    for link in soup.find_all('a'):
        href = link.get('href')
        if href:
            text = link.string
            if not text:
                text = ''
            text = re.sub('\\s+', ' ', text).strip()
            yield (parse.urljoin(root, link.get('href')), text)


def get_relevance(url):
    score = len(url.split('/'))
    score -= url.count('&') + url.count('?')
    score -= sum(1 if c.isdigit() else 0 for c in url)
    return score


def is_self_ref(root, url):
    root_data = parse.urlsplit(root)
    url_data = parse.urlsplit(url)
    return (root_data.scheme == url_data.scheme and 
            root_data.netloc == url_data.netloc and 
            root_data.path.rstrip('/') == url_data.path.rstrip('/'))


def parse_links_sorted(root, html):
    # TODO: implement
    links = list(parse_links(root, html))
    scores = []

    for l, t in links:
        score = get_relevance(l)
        scores.append((score, l, t))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [(l, t) for s, l, t in scores]


def get_links(url):
    res = request.urlopen(url)
    return list(parse_links_sorted(url, res.read()))


def get_nonlocal_links(url):
    '''Get a list of links on the page specificed by the url,
    but only keep non-local links and non self-references.
    Return a list of (link, title) pairs, just like get_links()'''

    # TODO: implement
    links = get_links(url)
    filtered = set()

    split = parse.urlsplit(url)
    host = split.hostname
    for l, t in links:
        l_split = parse.urlsplit(l)
        if not l_split.netloc or l_split.hostname == host: continue
        filtered.add((l, t))

    return list(filtered)


def crawl(root, wanted_content=[], within_domain=True):
    '''Crawl the url specified by `root`.
    `wanted_content` is a list of content types to crawl
    `within_domain` specifies whether the crawler should limit itself to the domain of `root`
    '''
    # TODO: implement

    queue = PriorityQueue()
    queue.put((0, root)) # give root highest priority

    visited = set()
    extracted = []

    ROOT_HOST = parse.urlsplit(root).hostname

    while not queue.empty():
        url = queue.get()[1]
        if url in visited: continue     # maintain efficiency with visited set
        try:
            req = request.urlopen(url)
            content_type = req.info().get_content_type()
            if content_type not in wanted_content: continue # skip pages of unwanted content
            html = req.read()

            visited.add(url)
            visitlog.debug(url)

            for ex in extract_information(url, html):
                extracted.append(ex)
                extractlog.debug(ex)

            # TODO: is this right? are self-ref pages ok or excluded? instructions are confusing
            if within_domain:
                # crawl links only in same domain if specified
                for link, title in get_links(url):
                    host = parse.urlsplit(link).hostname
                    if host == ROOT_HOST and not is_self_ref(url, link):
                        score = get_relevance(link)
                        queue.put((score, link))

            else:
                # ensure only enqueuing nonlocal links
                for link, title in get_nonlocal_links(url):
                    if not is_self_ref(url, link):
                        score = get_relevance(link)
                        queue.put((score, link))

        except Exception as e:
            print(e, url)

    return visited, extracted


def extract_information(address, html):
    '''Extract contact information from html, returning a list of (url, category, content) pairs,
    where category is one of PHONE, ADDRESS, EMAIL'''

    # TODO: implement
    results = []
    # phone numbers
    for match in re.findall('\\d\\d\\d-\\d\\d\\d-\\d\\d\\d\\d', str(html)):
        results.append((address, 'PHONE', match))
    # email addresses 
    for match in re.findall('[a-zA-Z\\d._-]+@[a-zA-Z\\d]+\\.[a-zA-Z]+', str(html)):
        results.append((address, 'EMAIL', match))
    # physical addresses
    for match in re.findall('[a-zA-Z]+(?:[ ][a-zA-Z]+)*, [a-zA-Z ]+[.]* \\d\\d\\d\\d\\d', str(html)):
        results.append((address, 'ADDRESS', match))

    return results


def writelines(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            print(d, file=fout)


def main():
    site = sys.argv[1]

    links = get_links(site)
    writelines('links.txt', links)

    nonlocal_links = get_nonlocal_links(site)
    writelines('nonlocal.txt', nonlocal_links)

    visited, extracted = crawl(site, 'text/html')
    writelines('visited.txt', visited)
    writelines('extracted.txt', extracted)


if __name__ == '__main__':
    main()