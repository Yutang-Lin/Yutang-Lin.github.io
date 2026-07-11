---
layout: page
permalink: /repositories/
title: repositories
description: A showcase of my GitHub repositories and contributions.
nav: true
nav_order: 4
---

{% if site.data.repositories.github_users %}

## GitHub profile

<div class="repositories repo-user-grid">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.liquid username=user %}
  {% endfor %}
</div>

{% endif %}

{% if site.data.repositories.github_repos %}

## Repositories

<section class="research-stats repo-stats reveal-on-scroll" aria-label="Aggregate repository statistics">
  <span class="stat">
    <span class="stat-value" data-repo-stat="stars" data-count="0">—</span>
    <span class="stat-label">Total Stars</span>
  </span>
  <span class="stat">
    <span class="stat-value" data-repo-stat="forks" data-count="0">—</span>
    <span class="stat-label">Total Forks</span>
  </span>
  <span class="stat">
    <span class="stat-value" data-repo-stat="repos" data-count="{{ site.data.repositories.github_repos | size }}">{{ site.data.repositories.github_repos | size }}</span>
    <span class="stat-label">Repositories</span>
  </span>
  <span class="stat-source">
    <i class="ti ti-brand-github"></i> live via GitHub
  </span>
</section>

<div class="repositories repo-grid">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}
