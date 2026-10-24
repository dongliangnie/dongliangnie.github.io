---
permalink: /
title: "Dongliang Nie"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

I am an undergraduate in Computer Science and Technology at Shanghai University (2023–2027).In 2027,  I will join Fudan University as a PhD student, under the supervision of Professor Hao Li.

My research interests lie in representation learning, world models, and large language model (LLM) post-training.


## Publications

{% for post in site.publications reversed %}
  {% include publication.html %}
{% endfor %}

## Education

<div class="edu-list">
  <div class="edu-item">
    <img class="edu-item__logo edu-item__logo--shu" src="{{ '/images/shu-emblem.png' | relative_url }}" alt="Shanghai University emblem">
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
