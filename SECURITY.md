# Security / Безопасность

## Reporting / Как сообщить

Please report vulnerabilities privately via GitHub: **Security → Report a vulnerability** in this repository.
Do not open a public issue for security problems. Include steps to reproduce and the affected version.

Об уязвимостях пишите приватно через GitHub: **Security → Report a vulnerability** в этом репозитории.
Не открывайте публичный issue. Приложите шаги воспроизведения и версию.

## Scope / Что считается уязвимостью

- The web UI (`python -m konveyer web`) is a local tool without authentication. It must only be reachable
  from the same machine: `[web] host` accepts loopback addresses only, and requests with a non-loopback
  `Host` header are rejected. Anything that lets another site or machine drive it is in scope.
- Leaking secrets (`DEEPSEEK_API_KEY`, `TG_BOT_TOKEN`, `HF_TOKEN`, cookies) to logs, job files or the web UI.
- Reading or sending files outside the job folder.

- Веб-интерфейс — локальный инструмент без пароля, доступ только с этого компьютера. Всё, что позволяет
  чужому сайту или компьютеру им управлять, — уязвимость.
- Утечка секретов в журнал, файлы задания или веб-интерфейс.
- Чтение или отправка файлов вне папки задания.

## Supported versions / Поддерживаемые версии

Only the latest release on `main` gets fixes. / Исправления — только для последней версии в `main`.
