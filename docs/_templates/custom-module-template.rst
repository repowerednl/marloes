{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Module Attributes

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree:
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree:
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree:
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block pydantic_models %}
{% if pydantic_models %}
.. rubric:: {{ _('Pydantic Models') }}

.. autosummary::
   :toctree:
   :template: custom-model-template.rst
   :nosignatures:
{% for item in pydantic_models %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block pydantic_settings %}
{% if pydantic_settings %}
.. rubric:: {{ _('Pydantic Settings') }}

.. autosummary::
   :toctree:
   :template: custom-settings-template.rst
   :nosignatures:
{% for item in pydantic_settings %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
