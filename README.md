# Dongliang Nie's Homepage

Personal academic website built with [Academic Pages](https://academicpages.github.io/) (Jekyll + GitHub Pages).

## 写博客流程

### 1. 新建文章

在 `_posts/` 下新建文件，文件名格式为 `YYYY-MM-DD-slug.md`，例如：

```text
_posts/2026-10-23-natural-gradient-descent-fisher-information.md
```

### 2. 添加 Front Matter

文件顶部必须有 YAML front matter，`layout` 已在 `_config.yml` 的 defaults 中默认设为 `single`，可省略：

```yaml
---
title: "文章标题"
date: 2026-10-23
categories:
  - 机器学习
tags:
  - 优化
  - 自然梯度
---
```

- `date` 决定文章排序与发布时间
- `categories` 参与生成 URL（当前 permalink 为 `/:categories/:title/`），多级分类会生成多级路径
- `tags` 用于标签归档

### 3. 正文

正文使用 Markdown 书写：

- 公式直接使用 `$...$`（行内）和 `$$...$$`（块级），站点已全局加载 MathJax
- 图片放在 `images/blog/` 下，使用根路径引用：

```markdown
![图片说明](/images/blog/example.png)
```

- 文件可放在 `files/` 下供下载，访问路径为 `/files/example.pdf`

### 4. 本地预览

1. Docker（推荐，镜像已存在）

~~~bash
cd /home/niedongliang/workspace/dongliangnie.github.io
docker compose up -d --build
docker compose run --rm --service-ports jekyll-site jekyll serve -H 0.0.0.0 -w --config _config.yml,_config_docker.yml
~~~

 然后打开 http://localhost:4000 。首次会自动 bundle install，之后启动很快。

2. 原生 Ruby

当前系统缺少 ruby-dev 头文件（eventmachine 等原生扩展编译失败），先安装再构建：

~~~bash
sudo apt-get install -y ruby-dev build-essential
export PATH="$HOME/.local/share/gem/ruby/3.0.0/bin:$PATH"
bundle install
bundle exec jekyll serve --livereload
~~~


### 5. 发布

```bash
git add _posts/ images/blog/
git commit -m "Add post: <标题>"
git push origin master
```

推送到 `master` 后 GitHub Actions 会自动构建并发布，文章地址为：

```text
https://dongliangnie.github.io/<category>/<slug>/
```

### 参见问题
执行以下命令启动 Jekyll 网站时，Docker 报错：
~~~bash
docker compose run --rm --service-ports jekyll-site \
  jekyll serve -H 0.0.0.0 -w \
  --config _config.yml,_config_docker.yml
~~~
错误信息：
~~~
failed to set up container networking:
Bind for 0.0.0.0:4000 failed: port is already allocated
~~~
新启动的 Jekyll 容器需要绑定宿主机的 4000 端口，但该端口已被其他进程或容器占用，导致端口资源冲突，容器无法启动。
解决方法：
检查 Docker 容器
~~~bash
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}"
~~~
如果发现其他容器占用了 4000 端口，确认其用途后停止对应容器：
~~~bash
docker stop <容器ID>
~~~
### 参考

- 草稿可先放在 `_drafts/`，本地预览时加 `--drafts` 参数
- 其他内容集合：`_publications/`、`_talks/`、`_teaching/`、`_portfolio/`
