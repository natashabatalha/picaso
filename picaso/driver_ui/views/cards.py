"""
Cards: the building blocks of the Spectrum & Retrieval Setup page.

A card is one panel with a form of fields (built from the session state) and,
optionally, an extra template below the form for buttons, plots and downloads.
Any change to a card's form posts it; the card applies the values and the
whole page is re-rendered (and morphed in place by htmx).
"""


class Card:
    name = ""
    title = ""
    extra = None  # template rendered below the form, e.g. "spectrum/extras/run.html"
    clears_previews = True  # editing this card makes the PT/chemistry/cloud previews stale

    def visible(self, sess):
        return True

    def fields(self, sess):
        """Field tree for the form, or None for a card without one."""
        return None

    def notes(self, sess):
        """[(level, message)] shown above the form; level is info, success, warning or error."""
        return []

    def update(self, sess, form):
        """Applies a submitted form. Returns {field name: error}."""
        errors = sess.apply(self.fields(sess), form)
        self.after_update(sess)
        return errors

    def after_update(self, sess):
        """Keeps derived state consistent after the user's edits."""


def card_views(cards, sess, store):
    """
    What the page template needs for each visible card. `store` is the page
    state holding action messages; they are shown once, like Flask's flash().
    """
    messages = store.pop("messages", {})
    return [
        {"card": card, "fields": card.fields(sess), "notes": card.notes(sess) + messages.get(card.name, [])}
        for card in cards if card.visible(sess)
    ]


def set_messages(store, card_name, messages):
    """Messages from an action (e.g. a failed run), shown on the card in the next render."""
    store.setdefault("messages", {})[card_name] = messages
