---
permalink: /
title: "Dongliang Nie"
author_profile: true
hide_title: true
redirect_from:
  - /about/
  - /about.html
---

<div class="hero">
  <h1 class="hero__title">Dongliang Nie</h1>
  <p class="hero__tagline">Undergraduate in Computer Science and Technology at Shanghai University (2023–2027). In 2027 I will join Fudan University as a Ph.D. student, under the supervision of Professor Hao Li. My research interests lie in representation learning, world models, and large language model (LLM) post-training.</p>
  <ul class="hero__interests">
    <li>Representation Learning</li>
    <li>World Models</li>
    <li>LLM Post-training</li>
  </ul>
  <div class="hero__actions">
    <a class="hero__btn hero__btn--primary" href="mailto:niedongliang1217@shu.edu.cn"><i class="fas fa-envelope" aria-hidden="true"></i>Email</a>
    <a class="hero__btn" href="https://scholar.google.com/citations?user=1VIUUqAAAAAJ&hl=en&authuser=1"><i class="ai ai-google-scholar" aria-hidden="true"></i>Google Scholar</a>
    <a class="hero__btn" href="https://github.com/dongliangnie"><i class="fab fa-github" aria-hidden="true"></i>GitHub</a>
  </div>
</div>

## Publications

{% for post in site.publications reversed %}
  {% include publication.html %}
{% endfor %}

## Education

<div class="edu-list">
  <div class="edu-item">
    <img class="edu-item__logo" src="{{ '/images/shu-emblem.png' | relative_url }}" alt="Shanghai University emblem">
    <div class="edu-item__body">
      <div class="edu-item__school">Shanghai University</div>
      <div class="edu-item__degree">B.Eng. in Computer Science and Technology</div>
    </div>
    <div class="edu-item__period">2023 – 2027</div>
  </div>

  <div class="edu-item">
    <img class="edu-item__logo" src="{{ '/images/fudan-emblem.png' | relative_url }}" alt="Fudan University emblem">
    <div class="edu-item__body">
      <div class="edu-item__school">Fudan University</div>
      <div class="edu-item__degree">Ph.D. in Computer Science</div>
    </div>
    <div class="edu-item__period">2027 – 2032 (expected)</div>
  </div>
</div>

## Awards

* ICPC 50th Asia Regional Contest (Nanjing) — Bronze
* CCPC 11th China Collegiate Programming Contest (Jinan) — Bronze

## Contact

* Email: [niedongliang1217@shu.edu.cn](mailto:niedongliang1217@shu.edu.cn)
* GitHub: [github.com/dongliangnie](https://github.com/dongliangnie)
